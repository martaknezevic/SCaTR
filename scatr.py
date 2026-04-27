"""
SCATR
===============================================================
Train a neural-network classifier on intermediate-layer embeddings extracted
from a language model, then evaluate it with Best-of-N and Weighted Majority
Vote metrics across all available train/test turn combinations.

Data format
-----------
Each dataset is stored under:
    parsed_data/<model>/<dataset>/turn<N>.pkl   (or the npy/feather layout)

Each row must contain:
  - problem_id       : identifier used to group rollouts for BoN/WMV
  - rollout_idx      : rollout index within a problem
  - is_correct       : binary ground-truth label (0/1)
  - <embed columns>  : embedding vectors; column names encode the layer index
                       and type, e.g. "intermediate_layer_12_final"

Embedding types (--type)
-----------------------------------
  mean          flat mean-pool vector per layer
  final         flat last-token vector per layer
  special       sequence of special-token vectors per layer (variable length)
  last_10       sequence of the last 10 token vectors per layer (fixed length)
  attn_weighted flat attention-weighted mean vector per layer
  all           sequence of every real token per layer (variable length)

For 'special', 'last_10', 'attn_weighted', and 'all' the row contains
multiple token vectors instead of a single flat vector. These are handled
by MultiTokenTransformerClassifier with sinusoidal positional encoding.
Variable-length types ('special', 'all') use a padding mask; sequences are
truncated to MAX_MULTI_TOKEN_SEQ_LEN (default 64) tokens.
For 'all', only a single --layer should be specified.

Model types (--model-type)
---------------------------
  transformer (default)
      Multiple --layer indices → TransformerClassifier over the sequence of
      layer embeddings (one position per layer, learned positional embeddings).
      A single --layer index falls back to an MLP.

  nn
      Multiple --layer indices → one MLP per layer, then a trained ensemble:
        (1) Per-layer  : each MLP evaluated independently (label: layer_N).
        (2) Best-layer : the single MLP with lowest validation loss.
        (3) Ensemble   : all per-layer MLPs frozen; a softmax weighting vector
                         is trained on the shared validation set via BCE.
      A single --layer index uses one MLP (label: default).

Hyperparameter optimisation (HPO)
----------------------------------
By default, Optuna TPE is run for 96 trials with MedianPruner early stopping.
Pass --grid to instead run a random search over 100 sampled configurations from
the same hyperparameter space; the configuration with the lowest validation loss
is kept.

Checkpointing
-------------
Trained models are saved to ./saved_models/ and results are appended to
./results/results_<model_type>_<model>_<train>_<test>_<embed_type>.txt.
If all expected checkpoints for a (turn, seed, fraction) combination already
exist on disk, training is skipped and the saved model is loaded directly.

Usage examples
--------------
# Single embed type, Transformer across layers 12 and 24:
python scatr.py --model ollmo7b --train math500 --test aime \
    --type final --layer 12,24

# Single embed type, per-layer MLP + ensemble:
python scatr.py --model ollmo7b --train math500 --test aime \
    --type final --layer 12,24 --model-type nn

# Multi-token type (special tokens), single layer:
python scatr.py --model ollmo7b --train math500 --test aime \
    --type special --layer 24

# Combined training set:
python scatr.py --model gptoss --train1 math500 --train2 kodcode \
    --test aime --type final --layer 24

# Random grid search instead of Optuna HPO, with multiple rollout counts for evaluation:
python scatr.py --model gptoss --train math500 --test aime \
    --type final --layer 24 --grid --rollouts

# Use only 20% of training data:
python scatr.py --model gptoss --train math500 --test aime \
    --type final --layer 24 --train-fractions .2

Key flags
---------
  --model          LM identifier; used to locate parsed_data/<model>/
  --train          single training dataset name
  --train1/--train2  combine two training datasets (cannot use with --train)
  --test           single test dataset name
  --test1/--test2  evaluate on two test datasets jointly (cannot use with --test)
  --balance        when combining datasets, subsample each to the same number
                   of problem_ids (requires --train1/--train2)
  --type           single embedding type (see list above)
  --layer          comma-separated layer indices for --type
  --model-type     'transformer' (default) or 'nn'
  --grid           random search instead of Optuna HPO
  --rollouts       evaluate at n_rollouts ∈ {2, 4, 8, 12, all} in one run
  --n-rollouts     evaluate at a fixed rollout count
  --train-fractions  one or more data fractions in (0, 1] (default: 1.0)
  --num-gpus       number of GPUs to use (default: 8)
  --gpu-ids        comma-separated GPU IDs to use (overrides --num-gpus)
  --sequential     run training tasks sequentially (default: parallel pool)
  --tasks-per-gpu  number of concurrent training tasks per GPU (default: 1)
  --profile        measure and report FLOPs/MACs/latency for each model
"""

import math
import time
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import os
import optuna
from collections import defaultdict
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fixed seed for 50-50 train/test splits when train==test dataset.
SPLIT_SEED = 0

# Embedding types that produce multiple token vectors per row (not a single flat vector)
MULTI_TOKEN_TYPES = {'special', 'last_10', 'attn_weighted', 'all'}

# Maximum sequence length for variable-length token types (special, all).
MAX_MULTI_TOKEN_SEQ_LEN = 64

def seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_embed_type(embedding_col):
    """Infer the embedding type from the column name(s)."""
    col = embedding_col[0] if isinstance(embedding_col, list) else embedding_col
    for suffix in ('_special', '_last_10', '_mean', '_final', '_attn_weighted', '_all'):
        if col.endswith(suffix):
            return suffix.lstrip('_')
    return None


def load_data(model_name, dataset_name, turn, columns=None):
    """
    Load a turn dataframe.
    If the npy/feather format exists, load from that (fast).
    columns: optional list of embedding column names to load.
             If None, load all columns.
             Metadata columns (meta.feather) are always loaded.
    Falls back to pkl if npy format not present.
    """
    base_dir = PARSED_DATA_DIR / model_name / dataset_name / f"turn{turn}"
    pkl_path = PARSED_DATA_DIR / model_name / dataset_name / f"turn{turn}.pkl"

    if (base_dir / 'meta.feather').exists():
        cols_to_load = [
            p.stem for p in sorted(base_dir.glob('*.npy'))
            if not p.stem.endswith('_offsets')
            and (columns is None or p.stem in columns)
        ]
        print(f"Loading {model_name}/{dataset_name}/turn{turn} from npy "
              f"({len(cols_to_load)} column(s): {cols_to_load[:3]}"
              f"{'...' if len(cols_to_load) > 3 else ''})")
        df = pd.read_feather(base_dir / 'meta.feather')
        for col in cols_to_load:
            npy_path     = base_dir / f'{col}.npy'
            offsets_path = base_dir / f'{col}_offsets.npy'
            if offsets_path.exists():
                flat    = np.load(npy_path)
                offsets = np.load(offsets_path)
                df[col] = [flat[offsets[i]:offsets[i+1]] for i in range(len(df))]
            else:
                df[col] = list(np.load(npy_path))
        return df

    print(f"Loading {model_name}/{dataset_name}/turn{turn} from pkl (npy not found)")
    return pd.read_pickle(pkl_path)


def load_data_combined(model_name, dataset_names, turn, columns=None):
    dfs = []
    for ds in dataset_names:
        df = load_data(model_name, ds, turn, columns=columns)
        df = df.copy()
        df['_source_dataset'] = ds
        df['problem_id'] = ds + '__' + df['problem_id'].astype(str)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def split_50_50(df, seed):
    seed_everywhere(seed)
    problem_ids = df['problem_id'].unique()
    np.random.shuffle(problem_ids)
    split_idx = len(problem_ids) // 2
    train_ids = set(problem_ids[:split_idx])
    test_ids = set(problem_ids[split_idx:])
    train_df = df[df['problem_id'].isin(train_ids)].copy()
    test_df = df[df['problem_id'].isin(test_ids)].copy()
    return train_df, test_df

def split_val(df, seed, val_ratio=0.25):
    seed_everywhere(seed)
    problem_ids = np.array(df['problem_id'].unique())
    shuffled_ids = np.random.permutation(problem_ids)
    val_size = int(len(shuffled_ids) * val_ratio)
    val_problem_ids = set(shuffled_ids[:val_size])
    train_problem_ids = set(shuffled_ids[val_size:])
    train_df = df[df['problem_id'].isin(train_problem_ids)].copy()
    val_df = df[df['problem_id'].isin(val_problem_ids)].copy()
    return train_df, val_df

def prepare_features(df, embedding_col):
    if isinstance(embedding_col, list):
        arrays = [np.stack(df[col].values) for col in embedding_col]
        X = np.stack(arrays, axis=1)
    else:
        X = np.stack(df[embedding_col].values)
    y = df['is_correct'].values
    return X, y


def prepare_features_multi_token(df, embedding_cols, embed_type,
                                  max_seq_len=MAX_MULTI_TOKEN_SEQ_LEN):
    if isinstance(embedding_cols, str):
        embedding_cols = [embedding_cols]

    col_arrays = [df[col].values for col in embedding_cols]
    n = len(df)

    all_samples = []
    for i in range(n):
        pieces = [col_arrays[j][i] for j in range(len(embedding_cols))]
        combined = np.concatenate(pieces, axis=0).astype(np.float32)
        all_samples.append(combined)

    if embed_type in ['last_10', 'attn_weighted']:
        X = np.stack(all_samples)
        return X, None

    else:
        if max_seq_len is not None:
            raw_max = max(s.shape[0] for s in all_samples)
            n_trunc = sum(1 for s in all_samples if s.shape[0] > max_seq_len)
            if n_trunc:
                print(f'    [prepare_features] Truncating {n_trunc}/{n} samples '
                      f'to max_seq_len={max_seq_len} (raw max was {raw_max} tokens); '
                      f'keeping last {max_seq_len} tokens (latter layers)')
            all_samples = [s[-max_seq_len:] if s.shape[0] > max_seq_len else s
                           for s in all_samples]

        max_len    = max(s.shape[0] for s in all_samples)
        hidden_dim = all_samples[0].shape[1]

        X    = np.zeros((n, max_len, hidden_dim), dtype=np.float32)
        mask = np.ones((n, max_len), dtype=bool)

        for i, s in enumerate(all_samples):
            seq_len = s.shape[0]
            X[i, :seq_len] = s
            mask[i, :seq_len] = False

        return X, mask


def get_hyperparameters_transformer(n_samples):
    if n_samples < 1000:
        return {
            'd_model_nhead_options': [(128, 4), (128, 8), (256, 4), (256, 8)],
            'num_encoder_layers_options': [1, 2],
            'dim_feedforward_options': [256, 512],
            'dropout_options': [0.3, 0.4, 0.5, 0.6],
            'lr_options': [1e-4, 5e-4, 1e-3],
            'weight_decay_options': [1e-3, 5e-3, 1e-2],
            'batch_size_options': [16, 32],
        }
    elif n_samples < 5000:
        return {
            'd_model_nhead_options': [(128, 4), (128, 8), (256, 4), (256, 8)],
            'num_encoder_layers_options': [1, 2, 3],
            'dim_feedforward_options': [256, 512, 1024],
            'dropout_options': [0.1, 0.2, 0.3, 0.4],
            'lr_options': [1e-4, 5e-4, 1e-3],
            'weight_decay_options': [1e-4, 1e-3, 5e-3],
            'batch_size_options': [32, 64, 128],
        }
    else:
        return {
            'd_model_nhead_options': [(256, 8), (512, 8), (512, 4), (256, 4)],
            'num_encoder_layers_options': [2, 3, 4],
            'dim_feedforward_options': [512, 1024, 2048],
            'dropout_options': [0.1, 0.2, 0.3],
            'lr_options': [1e-4, 5e-4, 1e-3],
            'weight_decay_options': [1e-5, 1e-4, 1e-3],
            'batch_size_options': [64, 128, 256],
        }


def get_hyperparameters_adaptive(n_samples, input_dim):
    if n_samples < 1000:
        return {
            'hidden_dims_options': [
                [512, 256],
                [512, 256, 128],
                [1024, 512],
                [1024, 512, 256],
            ],
            'lr_options': [1e-4, 5e-4, 1e-3],
            'dropout_options': [0.5, 0.6, 0.7],
            'input_dropout_options': [0.2, 0.3, 0.4],
            'batch_size_options': [16, 32],
            'weight_decay_options': [1e-3, 5e-3, 1e-2],
            'use_bn_options': [False],
        }
    elif n_samples < 5000:
        return {
            'hidden_dims_options': [
                [512, 256],
                [1024, 512],
                [512, 256, 128],
                [1024, 512, 256],
            ],
            'lr_options': [1e-4, 5e-4, 1e-3],
            'dropout_options': [0.4, 0.5, 0.6],
            'input_dropout_options': [0.1, 0.2, 0.3],
            'batch_size_options': [32, 64, 128],
            'weight_decay_options': [1e-4, 1e-3, 5e-3],
            'use_bn_options': [False, True],
        }
    else:
        return {
            'hidden_dims_options': [
                [1024, 512],
                [2048, 1024],
                [1024, 512, 256],
                [2048, 1024, 512],
            ],
            'lr_options': [1e-4, 5e-4, 1e-3],
            'dropout_options': [0.3, 0.4, 0.5],
            'input_dropout_options': [0.0, 0.1, 0.2],
            'batch_size_options': [64, 128, 256],
            'weight_decay_options': [1e-5, 1e-4, 1e-3],
            'use_bn_options': [False, True],
        }

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.5, input_dropout=0.2, use_bn=False):
        super().__init__()
        self.init_args = (input_dim, hidden_dims, dropout, input_dropout, use_bn)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.input_dropout_p = input_dropout
        self.use_bn = use_bn
        self.output_dim = 1
        self.input_dropout = nn.Dropout(input_dropout)
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_dropout(x)
        return self.network(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_layers_seq, d_model=128, nhead=4,
                 num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.init_args = (input_dim, num_layers_seq, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.input_dim = input_dim
        self.num_layers_seq = num_layers_seq
        self.d_model = d_model
        self.output_dim = 1

        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(num_layers_seq, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embedding(positions)
        x = self.input_dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class MultiTokenTransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4,
                 num_encoder_layers=2, dim_feedforward=256, dropout=0.1,
                 max_len=MAX_MULTI_TOKEN_SEQ_LEN):
        super().__init__()
        self.init_args = (input_dim, d_model, nhead, num_encoder_layers,
                          dim_feedforward, dropout, max_len)
        self.input_dim  = input_dim
        self.d_model    = d_model
        self.output_dim = 1

        self.input_proj    = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,
                                                  enable_nested_tensor=False)
        self.classifier  = nn.Linear(d_model, 1)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = x + self.pe[:x.size(1)]
        x = self.input_dropout(x)

        if src_key_padding_mask is not None:
            real = ~src_key_padding_mask
            x = x * real.unsqueeze(-1).float()

        x = self.transformer(x)

        if src_key_padding_mask is not None:
            denom = real.float().sum(dim=1, keepdim=True).clamp(min=1e-9)
            x     = (x * real.unsqueeze(-1).float()).sum(dim=1) / denom
        else:
            x = x.mean(dim=1)

        return self.classifier(x)


class EnsembleMLP(nn.Module):
    def __init__(self, models, embedding_cols):
        super().__init__()
        self.models         = nn.ModuleList(models)
        self.embedding_cols = embedding_cols
        self.log_weights    = nn.Parameter(torch.zeros(len(models)))
        self.input_dim      = models[0].input_dim
        self.output_dim     = 1
        self.init_args      = None

    def forward(self, x):
        weights = torch.softmax(self.log_weights, dim=0)
        probs   = torch.stack(
            [torch.sigmoid(model(x[col]))
             for model, col in zip(self.models, self.embedding_cols)],
            dim=-1
        )
        return (probs * weights).sum(dim=-1)


def train_ensemble_weights(trained, val_df, seed, device):
    seed_everywhere(seed)
    cols   = [col   for col, _,  _     in trained]
    models = [model for _,   _,  model in trained]

    for model in models:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

    ensemble = EnsembleMLP(models, cols).to(device)

    x_val_dict = {
        col: torch.FloatTensor(np.stack(val_df[col].values)).to(device)
        for col in cols
    }
    y_val_t = torch.FloatTensor(val_df['is_correct'].values).unsqueeze(1).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam([ensemble.log_weights], lr=1e-2, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss    = float('inf')
    best_log_weights = None
    patience_counter = 0
    ensemble_steps   = 0

    for epoch in range(300):
        ensemble.train()
        optimizer.zero_grad()
        loss = criterion(ensemble(x_val_dict), y_val_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([ensemble.log_weights], 1.0)
        optimizer.step()
        ensemble_steps += 1

        ensemble.eval()
        with torch.no_grad():
            val_loss = criterion(ensemble(x_val_dict), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_log_weights = ensemble.log_weights.data.clone()
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break

    with torch.no_grad():
        ensemble.log_weights.copy_(best_log_weights)

    with torch.no_grad():
        weights = torch.softmax(ensemble.log_weights, dim=0).cpu().numpy()
    for col, w in zip(cols, weights):
        print(f"      [ensemble] trained weight {col}: {w:.4f}")
    print(f"      [ensemble] best val BCE loss = {best_val_loss:.4f}")
    print(f"      [ensemble] gradient steps (epochs) = {ensemble_steps}")

    return ensemble, ensemble_steps


def _train_one_config_mlp(model, X_train_t, y_train_t, X_val_t, y_val_t,
                           criterion, lr, wd, batch_size, seed, config_idx):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = None
    steps            = 0
    n                = len(X_train_t)

    for epoch in range(100):
        seed_everywhere(seed + config_idx)
        model.train()
        indices = torch.randperm(n, device=X_train_t.device)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            optimizer.zero_grad()
            criterion(model(X_train_t[batch_idx]), y_train_t[batch_idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    return best_val_loss, best_state, steps


def train_single_mlp(X_train, y_train, X_val, y_val, seed, device, pos_weight_tensor,
                     use_grid=False):
    seed_everywhere(seed)
    n_samples = len(X_train)
    input_dim = X_train.shape[1]
    hp = get_hyperparameters_adaptive(n_samples, input_dim)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train.copy()).unsqueeze(1).to(device)
    X_val_t   = torch.FloatTensor(X_val).to(device)
    y_val_t   = torch.FloatTensor(y_val.copy()).unsqueeze(1).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    if use_grid:
        rng         = np.random.RandomState(seed)
        total_steps = 0
        best_val_loss_g = float('inf')
        best_model_g    = None

        print(f"    Grid random search (100 configs) …")
        for config_idx in range(100):
            hidden_dims   = hp['hidden_dims_options'][rng.randint(len(hp['hidden_dims_options']))]
            dropout       = float(rng.choice(hp['dropout_options']))
            input_dropout = float(rng.choice(hp['input_dropout_options']))
            use_bn        = bool(rng.choice(hp['use_bn_options']))
            lr            = float(rng.choice(hp['lr_options']))
            wd            = float(rng.choice(hp['weight_decay_options']))
            batch_size    = int(rng.choice(hp['batch_size_options']))
            batch_size    = max(min(batch_size, n_samples // 4), 32)

            model = MLPClassifier(input_dim, hidden_dims, dropout, input_dropout, use_bn).to(device)
            val_loss, best_state, steps = _train_one_config_mlp(
                model, X_train_t, y_train_t, X_val_t, y_val_t,
                criterion, lr, wd, batch_size, seed, config_idx
            )
            total_steps += steps

            if val_loss < best_val_loss_g:
                best_val_loss_g = val_loss
                best_model_g = MLPClassifier(
                    input_dim, hidden_dims, dropout, input_dropout, use_bn
                ).to(device)
                best_model_g.load_state_dict(best_state)

        print(f"    Grid search: best val loss = {best_val_loss_g:.4f} (100 configs)")
        return best_val_loss_g, best_model_g, total_steps

    total_steps = 0

    def make_model(trial):
        hidden_dims_idx = trial.suggest_categorical('hidden_dims_idx', list(range(len(hp['hidden_dims_options']))))
        hidden_dims     = hp['hidden_dims_options'][hidden_dims_idx]
        dropout         = trial.suggest_categorical('dropout', hp['dropout_options'])
        input_dropout   = trial.suggest_categorical('input_dropout', hp['input_dropout_options'])
        use_bn          = trial.suggest_categorical('use_bn', hp['use_bn_options'])
        return MLPClassifier(input_dim, hidden_dims, dropout, input_dropout, use_bn).to(device)

    def objective(trial):
        nonlocal total_steps
        seed_everywhere(seed)
        model      = make_model(trial)
        lr         = trial.suggest_categorical('lr', hp['lr_options'])
        wd         = trial.suggest_categorical('weight_decay', hp['weight_decay_options'])
        batch_size = trial.suggest_categorical('batch_size', hp['batch_size_options'])
        batch_size = max(min(batch_size, n_samples // 4), 32)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss    = float('inf')
        patience_counter = 0
        best_state       = None

        for epoch in range(100):
            seed_everywhere(seed)
            model.train()
            indices = torch.randperm(len(X_train_t), device=device)
            for start in range(0, len(X_train_t), batch_size):
                batch_idx = indices[start:start + batch_size]
                optimizer.zero_grad()
                criterion(model(X_train_t[batch_idx]), y_train_t[batch_idx]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_steps += 1

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            scheduler.step(val_loss)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        trial.set_user_attr('best_state', best_state)
        return best_val_loss

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=96, n_jobs=1, show_progress_bar=False)

    best_trial = study.best_trial
    final_model = make_model(best_trial)
    final_model.load_state_dict(best_trial.user_attrs['best_state'])
    return study.best_value, final_model, total_steps


def _train_one_config_transformer(model, X_train_t, y_train_t, X_val_t, y_val_t,
                                   criterion, lr, wd, batch_size, seed, config_idx,
                                   mask_train_t=None, mask_val_t=None):
    device   = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = None
    steps            = 0
    n                = len(X_train_t)
    has_mask         = mask_train_t is not None

    for epoch in range(100):
        seed_everywhere(seed + config_idx)
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx  = perm[start:start + batch_size]
            batch_X    = X_train_t[batch_idx].to(device)
            batch_y    = y_train_t[batch_idx].to(device)
            batch_mask = mask_train_t[batch_idx].to(device) if has_mask else None
            optimizer.zero_grad()
            if has_mask:
                criterion(model(batch_X, src_key_padding_mask=batch_mask), batch_y).backward()
            else:
                criterion(model(batch_X), batch_y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1

        model.eval()
        val_chunk    = max(batch_size, 64)
        val_loss_sum = 0.0
        val_n        = len(X_val_t)
        with torch.no_grad():
            for vstart in range(0, val_n, val_chunk):
                vX   = X_val_t[vstart:vstart + val_chunk].to(device)
                vy   = y_val_t[vstart:vstart + val_chunk].to(device)
                vmsk = mask_val_t[vstart:vstart + val_chunk].to(device) if has_mask else None
                chunk_loss = criterion(
                    model(vX, src_key_padding_mask=vmsk) if has_mask else model(vX), vy
                )
                val_loss_sum += chunk_loss.item() * len(vX)
        val_loss = val_loss_sum / val_n
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    return best_val_loss, best_state, steps


def train_model(X_train, y_train, X_val, y_val, seed, gpu_id=0, train_turn=None,
                use_grid=False):
    seed_everywhere(seed)
    torch.backends.cudnn.benchmark = True
    if train_turn is not None:
        print(f"Starting training for Train Turn {train_turn}, Seed {seed} (GPU {gpu_id})")
    else:
        print(f"Starting training for Seed {seed}")
    device = torch.device(f'cuda:{gpu_id}')
    torch.set_float32_matmul_precision('high')

    unique, counts = np.unique(y_train, return_counts=True)
    pos_weight = counts[0] / counts[1] if unique[1] == 1 else counts[1] / counts[0]
    pos_weight_tensor = torch.FloatTensor([pos_weight]).to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train.copy()).unsqueeze(1).to(device)
    X_val_t   = torch.FloatTensor(X_val).to(device)
    y_val_t   = torch.FloatTensor(y_val.copy()).unsqueeze(1).to(device)

    n_samples      = len(X_train)
    is_transformer = X_train.ndim == 3

    print(f"    Mode: {'Transformer' if is_transformer else 'MLP'} | "
          f"Shape: {X_train.shape} | Samples: {n_samples}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    total_steps = 0

    if is_transformer:
        num_layers_seq = X_train.shape[1]
        input_dim      = X_train.shape[2]
        hp             = get_hyperparameters_transformer(n_samples)

        def make_model_transformer(cfg):
            d_model, nhead = cfg['d_model_nhead']
            return TransformerClassifier(
                input_dim, num_layers_seq,
                d_model, nhead, cfg['num_encoder_layers'],
                cfg['dim_feedforward'], cfg['dropout']
            ).to(device)

        if use_grid:
            rng = np.random.RandomState(seed)
            best_val_loss_g = float('inf')
            best_cfg_g      = None
            best_state_g    = None

            print(f"    Grid random search (100 configs) …")
            for config_idx in range(100):
                cfg = {
                    'd_model_nhead':      hp['d_model_nhead_options'][rng.randint(len(hp['d_model_nhead_options']))],
                    'num_encoder_layers': int(rng.choice(hp['num_encoder_layers_options'])),
                    'dim_feedforward':    int(rng.choice(hp['dim_feedforward_options'])),
                    'dropout':            float(rng.choice(hp['dropout_options'])),
                }
                lr         = float(rng.choice(hp['lr_options']))
                wd         = float(rng.choice(hp['weight_decay_options']))
                batch_size = int(rng.choice(hp['batch_size_options']))
                batch_size = max(min(batch_size, n_samples // 4), 32)

                model = make_model_transformer(cfg)
                val_loss, best_state, steps = _train_one_config_transformer(
                    model, X_train_t, y_train_t, X_val_t, y_val_t,
                    criterion, lr, wd, batch_size, seed, config_idx
                )
                total_steps += steps

                if val_loss < best_val_loss_g:
                    best_val_loss_g = val_loss
                    best_cfg_g      = cfg
                    best_state_g    = best_state

            print(f"    Grid search: best val loss = {best_val_loss_g:.4f} (100 configs)")
            final_model = make_model_transformer(best_cfg_g)
            final_model.load_state_dict(best_state_g)
            return final_model, device, total_steps

        def make_model(trial):
            d_model_nhead_idx = trial.suggest_categorical('d_model_nhead_idx', list(range(len(hp['d_model_nhead_options']))))
            d_model, nhead = hp['d_model_nhead_options'][d_model_nhead_idx]
            num_enc        = trial.suggest_categorical('num_encoder_layers', hp['num_encoder_layers_options'])
            dim_ff         = trial.suggest_categorical('dim_feedforward', hp['dim_feedforward_options'])
            dropout        = trial.suggest_categorical('dropout', hp['dropout_options'])
            return TransformerClassifier(
                input_dim, num_layers_seq, d_model, nhead, num_enc, dim_ff, dropout
            ).to(device)

    else:
        input_dim = X_train.shape[1]
        hp        = get_hyperparameters_adaptive(n_samples, input_dim)

        def make_model_mlp(cfg):
            return MLPClassifier(
                input_dim,
                cfg['hidden_dims'], cfg['dropout'],
                cfg['input_dropout'], cfg['use_bn']
            ).to(device)

        if use_grid:
            rng = np.random.RandomState(seed)
            best_val_loss_g = float('inf')
            best_cfg_g      = None
            best_state_g    = None

            print(f"    Grid random search (100 configs) …")
            for config_idx in range(100):
                cfg = {
                    'hidden_dims':   hp['hidden_dims_options'][rng.randint(len(hp['hidden_dims_options']))],
                    'dropout':       float(rng.choice(hp['dropout_options'])),
                    'input_dropout': float(rng.choice(hp['input_dropout_options'])),
                    'use_bn':        bool(rng.choice(hp['use_bn_options'])),
                }
                lr         = float(rng.choice(hp['lr_options']))
                wd         = float(rng.choice(hp['weight_decay_options']))
                batch_size = int(rng.choice(hp['batch_size_options']))
                batch_size = max(min(batch_size, n_samples // 4), 32)

                model = make_model_mlp(cfg)
                val_loss, best_state, steps = _train_one_config_transformer(
                    model, X_train_t, y_train_t, X_val_t, y_val_t,
                    criterion, lr, wd, batch_size, seed, config_idx
                )
                total_steps += steps

                if val_loss < best_val_loss_g:
                    best_val_loss_g = val_loss
                    best_cfg_g      = cfg
                    best_state_g    = best_state

            print(f"    Grid search: best val loss = {best_val_loss_g:.4f} (100 configs)")
            final_model = make_model_mlp(best_cfg_g)
            final_model.load_state_dict(best_state_g)
            return final_model, device, total_steps

        def make_model(trial):
            hidden_dims_idx = trial.suggest_categorical('hidden_dims_idx', list(range(len(hp['hidden_dims_options']))))
            hidden_dims     = hp['hidden_dims_options'][hidden_dims_idx]
            dropout         = trial.suggest_categorical('dropout', hp['dropout_options'])
            input_dropout   = trial.suggest_categorical('input_dropout', hp['input_dropout_options'])
            use_bn          = trial.suggest_categorical('use_bn', hp['use_bn_options'])
            return MLPClassifier(input_dim, hidden_dims, dropout, input_dropout, use_bn).to(device)

    print(f"    Optuna hyperparameter search (96 trials, MedianPruner)")

    def objective(trial):
        nonlocal total_steps
        seed_everywhere(seed)
        model      = make_model(trial)
        lr         = trial.suggest_categorical('lr', hp['lr_options'])
        wd         = trial.suggest_categorical('weight_decay', hp['weight_decay_options'])
        batch_size = trial.suggest_categorical('batch_size', hp['batch_size_options'])
        batch_size = min(batch_size, n_samples // 4)
        batch_size = max(batch_size, 32)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_loss    = float('inf')
        patience_counter = 0
        best_state       = None

        for epoch in range(100):
            seed_everywhere(seed)
            model.train()
            indices = torch.randperm(len(X_train_t), device=device)
            for start in range(0, len(X_train_t), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_X   = X_train_t[batch_idx]
                batch_y   = y_train_t[batch_idx]
                optimizer.zero_grad()
                criterion(model(batch_X), batch_y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_steps += 1

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            scheduler.step(val_loss)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        trial.set_user_attr('best_state', best_state)
        return best_val_loss

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=96, n_jobs=1, show_progress_bar=False)

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"    Optuna: best val loss = {study.best_value:.4f} "
          f"(trial {study.best_trial.number}, {n_pruned}/96 pruned)")

    best_trial  = study.best_trial
    final_model = make_model(best_trial)
    final_model.load_state_dict(best_trial.user_attrs['best_state'])

    return final_model, device, total_steps


def train_model_multi_token(X_train, y_train, X_val, y_val, seed, gpu_id=0,
                             mask_train=None, mask_val=None, train_turn=None,
                             use_grid=False):
    seed_everywhere(seed)
    if train_turn is not None:
        print(f"Starting multi-token training for Train Turn {train_turn}, Seed {seed} (GPU {gpu_id})")
    else:
        print(f"Starting multi-token training for Seed {seed}")

    device = torch.device(f'cuda:0')
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    n_samples  = len(X_train)
    input_dim  = X_train.shape[2]
    seq_len    = X_train.shape[1]
    has_mask   = mask_train is not None

    print(f"    Mode: MultiTokenTransformer | Shape: {X_train.shape} | "
          f"Samples: {n_samples} | Padding mask: {has_mask}")

    unique, counts = np.unique(y_train, return_counts=True)
    pos_weight = counts[0] / counts[1] if unique[1] == 1 else counts[1] / counts[0]
    pos_weight_tensor = torch.FloatTensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train.copy()).unsqueeze(1)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.FloatTensor(y_val.copy()).unsqueeze(1)

    mask_train_t = torch.BoolTensor(mask_train) if has_mask else None
    mask_val_t   = torch.BoolTensor(mask_val)   if has_mask else None

    hp = get_hyperparameters_transformer(n_samples)

    total_steps = 0

    def make_model_cfg(cfg):
        d_model, nhead = cfg['d_model_nhead']
        return MultiTokenTransformerClassifier(
            input_dim, d_model, nhead,
            cfg['num_encoder_layers'], cfg['dim_feedforward'], cfg['dropout'],
            max_len=MAX_MULTI_TOKEN_SEQ_LEN,
        ).to(device)

    if use_grid:
        rng = np.random.RandomState(seed)
        best_val_loss_g = float('inf')
        best_cfg_g      = None
        best_state_g    = None

        print(f"    Grid random search (100 configs) …")
        for config_idx in range(100):
            cfg = {
                'd_model_nhead':      hp['d_model_nhead_options'][rng.randint(len(hp['d_model_nhead_options']))],
                'num_encoder_layers': int(rng.choice(hp['num_encoder_layers_options'])),
                'dim_feedforward':    int(rng.choice(hp['dim_feedforward_options'])),
                'dropout':            float(rng.choice(hp['dropout_options'])),
            }
            lr         = float(rng.choice(hp['lr_options']))
            wd         = float(rng.choice(hp['weight_decay_options']))
            batch_size = int(rng.choice(hp['batch_size_options']))
            batch_size = max(min(batch_size, n_samples // 4), 32)

            model = make_model_cfg(cfg)
            val_loss, best_state, steps = _train_one_config_transformer(
                model, X_train_t, y_train_t, X_val_t, y_val_t,
                criterion, lr, wd, batch_size, seed, config_idx,
                mask_train_t=mask_train_t, mask_val_t=mask_val_t
            )
            total_steps += steps

            if val_loss < best_val_loss_g:
                best_val_loss_g = val_loss
                best_cfg_g      = cfg
                best_state_g    = best_state

        print(f"    Grid search: best val loss = {best_val_loss_g:.4f} (100 configs)")
        final_model = make_model_cfg(best_cfg_g)
        final_model.load_state_dict(best_state_g)
        return final_model, device, total_steps

    def make_model(trial):
        d_model_nhead_idx = trial.suggest_categorical(
            'd_model_nhead_idx', list(range(len(hp['d_model_nhead_options'])))
        )
        d_model, nhead = hp['d_model_nhead_options'][d_model_nhead_idx]
        num_enc = trial.suggest_categorical('num_encoder_layers', hp['num_encoder_layers_options'])
        dim_ff  = trial.suggest_categorical('dim_feedforward',    hp['dim_feedforward_options'])
        dropout = trial.suggest_categorical('dropout',            hp['dropout_options'])
        return MultiTokenTransformerClassifier(
            input_dim, d_model, nhead, num_enc, dim_ff, dropout,
            max_len=MAX_MULTI_TOKEN_SEQ_LEN,
        ).to(device)

    print(f"    Optuna hyperparameter search (96 trials, MedianPruner)")

    def objective(trial):
        nonlocal total_steps
        seed_everywhere(seed)
        model      = make_model(trial)
        lr         = trial.suggest_categorical('lr',            hp['lr_options'])
        wd         = trial.suggest_categorical('weight_decay',  hp['weight_decay_options'])
        batch_size = trial.suggest_categorical('batch_size',    hp['batch_size_options'])
        batch_size = max(min(batch_size, n_samples // 4), 32)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_val_loss    = float('inf')
        patience_counter = 0
        best_state       = None

        for epoch in range(100):
            seed_everywhere(seed)
            model.train()
            perm = torch.randperm(len(X_train_t))
            for start in range(0, len(X_train_t), batch_size):
                batch_idx  = perm[start:start + batch_size]
                batch_X    = X_train_t[batch_idx].to(device)
                batch_y    = y_train_t[batch_idx].to(device)
                batch_mask = mask_train_t[batch_idx].to(device) if mask_train_t is not None else None
                optimizer.zero_grad()
                criterion(model(batch_X, src_key_padding_mask=batch_mask), batch_y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_steps += 1

            model.eval()
            val_chunk    = max(batch_size, 64)
            val_loss_sum = 0.0
            val_n        = len(X_val_t)
            with torch.no_grad():
                for vstart in range(0, val_n, val_chunk):
                    vX   = X_val_t[vstart:vstart + val_chunk].to(device)
                    vy   = y_val_t[vstart:vstart + val_chunk].to(device)
                    vmsk = mask_val_t[vstart:vstart + val_chunk].to(device) if mask_val_t is not None else None
                    chunk_loss = criterion(model(vX, src_key_padding_mask=vmsk), vy)
                    val_loss_sum += chunk_loss.item() * len(vX)
            val_loss = val_loss_sum / val_n
            scheduler.step(val_loss)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        trial.set_user_attr('best_state', best_state)
        return best_val_loss

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=30),
    )
    study.optimize(objective, n_trials=96, n_jobs=1, show_progress_bar=False)

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"    Optuna multi-token: best val loss = {study.best_value:.4f} "
          f"(trial {study.best_trial.number}, {n_pruned}/96 pruned)")

    best_trial  = study.best_trial
    final_model = make_model(best_trial)
    final_model.load_state_dict(best_trial.user_attrs['best_state'])
    return final_model, device, total_steps


def evaluate_best_of_n(df, predictions):
    df_eval = df.copy()
    df_eval['pred'] = predictions
    best_indices = df_eval.groupby('problem_id')['pred'].idxmax()
    best_rows = df_eval.loc[best_indices]
    accuracy = best_rows['is_correct'].mean()
    return accuracy

def evaluate_weighted_majority(df, predictions):
    df_eval = df.copy()
    df_eval['pred'] = predictions
    correct_count = 0
    total_count = 0
    for problem_id, group in df_eval.groupby('problem_id'):
        answer_scores = group.groupby('extracted_answer')['pred'].sum()
        best_answer = answer_scores.idxmax()
        answer_rows = group[group['extracted_answer'] == best_answer]
        is_correct = answer_rows.iloc[0]['is_correct']
        if is_correct:
            correct_count += 1
        total_count += 1
    accuracy = correct_count / total_count
    return accuracy

def drop_nan_embedding_rows(df, embedding_col):
    cols = embedding_col if isinstance(embedding_col, list) else [embedding_col]
    mask = pd.Series(True, index=df.index)
    for col in cols:
        mask &= df[col].apply(lambda arr: not np.any(np.isnan(np.asarray(arr).astype(np.float32))))
    return df[mask].copy()

def get_available_turns(model_name, dataset):
    base_dir = PARSED_DATA_DIR / model_name / dataset
    turns = []
    for turn in range(1, 10):
        file = base_dir / f"turn{turn}.pkl"
        if file.exists():
            turns.append(turn)
    return turns


def make_combined_train_name(train1, train2):
    return f"{train1}_{train2}_combined"


def get_save_filename(model_name, train_dataset, train_turn, seed, train_fraction,
                      embedding_col, effective_model_type, label, same_split=False):
    frac_str = "" if train_fraction == 1.0 else f"_frac{train_fraction}"
    same_str = "_same" if same_split else ""
    if isinstance(embedding_col, list):
        embed_type_str = infer_embed_type(embedding_col)
        layer_tag = "layers_" + "_".join(
            c.split("_layer_")[1].split("_")[0] for c in embedding_col
        ) + f"_{embed_type_str}"
    else:
        layer_tag = embedding_col
    label_tag = f"_{label}" if label != 'default' else ""
    return (
        f"model_{model_name}_{train_dataset}{same_str}_turn{train_turn}"
        f"_seed{seed}{frac_str}_{layer_tag}_{effective_model_type}{label_tag}.pt"
    )


def load_model_from_checkpoint(path, device, ckpt=None):
    if ckpt is None:
        ckpt = torch.load(path, map_location=device)
    meta = ckpt["model_meta"]
    class_name = meta["model_class"]
    init_args  = meta["init_args"]
    eff_col    = meta["training_info"].get("effective_embedding_col",
                                           meta["training_info"]["embedding_col"])

    _CLS = {
        "MLPClassifier":                   MLPClassifier,
        "TransformerClassifier":           TransformerClassifier,
        "MultiTokenTransformerClassifier": MultiTokenTransformerClassifier,
    }

    if class_name == "EnsembleMLP":
        members_info = meta["training_info"].get("ensemble_members", [])
        members = []
        for m_info in members_info:
            m_cls = _CLS[m_info["model_class"]]
            m = m_cls(*m_info["init_args"]).to(device)
            members.append(m)
        cols  = eff_col if isinstance(eff_col, list) else [eff_col]
        model = EnsembleMLP(members, cols).to(device)
    else:
        model = _CLS[class_name](*init_args).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, eff_col


def _run_multi_token_inference(model, df, embedding_col, embed_type, device, chunk_size=256):
    cols = embedding_col if isinstance(embedding_col, list) else [embedding_col]
    all_preds = []
    model.eval()
    n = len(df)

    X_full, mask_full = prepare_features_multi_token(df, cols, embed_type)
    X_full_t    = torch.FloatTensor(X_full)
    mask_full_t = torch.BoolTensor(mask_full) if mask_full is not None else None

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            chunk_X    = X_full_t[start:start + chunk_size].to(device)
            chunk_mask = (mask_full_t[start:start + chunk_size].to(device)
                          if mask_full_t is not None else None)
            preds = torch.sigmoid(
                model(chunk_X, src_key_padding_mask=chunk_mask)
            ).cpu().numpy().flatten()
            all_preds.append(preds)

    return np.concatenate(all_preds)


def profile_model_pass(model, embedding_col, df, embed_type, device, batch_size=1, tag=""):
    try:
        from thop import profile as thop_profile
        from thop import clever_format
    except ImportError:
        print("  [profile] thop not installed — skipping.")
        return None

    import time
    import copy

    sample_df    = df.iloc[:batch_size].copy()
    actual_batch = len(sample_df)
    model.eval()

    if embed_type in MULTI_TOKEN_TYPES:
        cols   = embedding_col if isinstance(embedding_col, list) else [embedding_col]
        X, mask = prepare_features_multi_token(sample_df, cols, embed_type)
        X_t    = torch.FloatTensor(X).to(device)
        mask_t = torch.BoolTensor(mask).to(device) if mask is not None else None
        thop_inputs = (X_t,)
    elif isinstance(model, EnsembleMLP):
        cols   = embedding_col if isinstance(embedding_col, list) else [embedding_col]
        x_dict = {
            col: torch.FloatTensor(np.stack(sample_df[col].values)).to(device)
            for col in cols
        }
        thop_inputs = (x_dict,)
        mask_t = None
    else:
        X, _   = prepare_features(sample_df, embedding_col)
        X_t    = torch.FloatTensor(X).to(device)
        thop_inputs = (X_t,)
        mask_t = None

    model_copy = copy.deepcopy(model)

    if embed_type in MULTI_TOKEN_TYPES and mask_t is not None:
        class _MaskedWrapper(nn.Module):
            def __init__(self, m, msk):
                super().__init__()
                self.m   = m
                self.msk = msk
            def forward(self, x):
                return self.m(x, src_key_padding_mask=self.msk)
        profiled_model = _MaskedWrapper(model_copy, mask_t)
    else:
        profiled_model = model_copy

    with torch.no_grad():
        macs_raw, params_raw = thop_profile(
            profiled_model, inputs=thop_inputs, verbose=False
        )
    del model_copy, profiled_model

    flops_raw = 2 * macs_raw
    macs_str,  params_str = clever_format([macs_raw,  params_raw], "%.3f")
    flops_str, _          = clever_format([flops_raw, params_raw], "%.3f")

    N_WARMUP = 5
    N_TIMED  = 20

    def _forward():
        with torch.no_grad():
            if embed_type in MULTI_TOKEN_TYPES:
                _ = model(X_t, src_key_padding_mask=mask_t)
            elif isinstance(model, EnsembleMLP):
                _ = model(x_dict)
            else:
                _ = model(X_t)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for _ in range(N_WARMUP):
        _forward()

    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        _forward()
    duration_s = (time.perf_counter() - t0) / N_TIMED

    return {
        'tag':                   tag,
        'batch_size':            actual_batch,
        'macs_total':            macs_str,
        'macs_per_sample':       f"{macs_raw / actual_batch:.4e}",
        'macs_raw':              macs_raw,
        'flops_total':           flops_str,
        'flops_per_sample':      f"{flops_raw / actual_batch:.4e}",
        'flops_raw':             flops_raw,
        'params':                params_str,
        'latency_total_ms':      f"{duration_s * 1000:.2f}",
        'latency_per_sample_ms': f"{duration_s * 1000 / actual_batch:.4f}",
    }


def process_train_seed(args_tuple):
    (model_name, train_turn, seed, gpu_id,
     train_dataset, test_dataset, embedding_col,
     gpu_ids_str, train_fraction, model_type, profile, use_grid,
     train_datasets_list, balance, load_columns) = args_tuple

    global PARSED_DATA_DIR
    PARSED_DATA_DIR = Path("./parsed_data")

    gpu_ids_list  = [int(x) for x in gpu_ids_str.split(',')]
    actual_gpu_id = gpu_ids_list[gpu_id]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(actual_gpu_id)
    seed_everywhere(seed)
    _wall_start = None
    try:
        train_val_accs    = None
        total_train_steps = 0
        per_layer_steps   = {}
        ensemble_steps    = 0
        per_layer_params  = {}

        print(f"Initiating training for Train Turn {train_turn}, Seed {seed} (GPU {gpu_id})")
        profile_results = {}

        _save_dir_check = Path("./saved_models")
        _same_split_check = (train_datasets_list is None) and (train_dataset == test_dataset)
        _embed_type_check = infer_embed_type(embedding_col)
        if _embed_type_check in MULTI_TOKEN_TYPES:
            _eff_model_type_check = 'multi_token_transformer'
        else:
            _eff_model_type_check = model_type if isinstance(embedding_col, list) else 'nn'

        def _expected_labels_check():
            if _embed_type_check in MULTI_TOKEN_TYPES:
                return [('default', embedding_col)]
            elif _eff_model_type_check == 'nn' and isinstance(embedding_col, list):
                per_layer = [
                    (f"layer_{c.split('_layer_')[1].split('_')[0]}", c)
                    for c in embedding_col
                ]
                return per_layer + [('best_layer', embedding_col), ('ensemble', embedding_col)]
            else:
                return [('default', embedding_col)]

        _expected = _expected_labels_check()
        _all_ckpts_present = all(
            (_save_dir_check / get_save_filename(
                model_name, train_dataset, train_turn, seed, train_fraction,
                tag_col, _eff_model_type_check, label,
                same_split=_same_split_check
            )).exists()
            for label, tag_col in _expected
        )

        if _all_ckpts_present:
            print(f"  [skip] All checkpoints found for Turn {train_turn} Seed {seed} — "
                  f"loading from disk, skipping training.")
            device = torch.device('cuda:0')
            loaded_entries = []
            for label, tag_col in _expected:
                ckpt_path = _save_dir_check / get_save_filename(
                    model_name, train_dataset, train_turn, seed, train_fraction,
                    tag_col, _eff_model_type_check, label,
                    same_split=_same_split_check
                )
                ckpt_loaded = torch.load(ckpt_path, map_location=device)
                model_loaded, eff_col_loaded = load_model_from_checkpoint(ckpt_path, device, ckpt=ckpt_loaded)
                loaded_entries.append((model_loaded, eff_col_loaded, label))
                t_info = ckpt_loaded["model_meta"]["training_info"]
                if not total_train_steps:
                    total_train_steps = t_info.get("total_train_steps", 0) or 0
                pls = t_info.get("per_layer_steps")
                if pls and not per_layer_steps:
                    per_layer_steps = pls
                es = t_info.get("ensemble_steps")
                if es and not ensemble_steps:
                    ensemble_steps = es
                plp = t_info.get("per_layer_params")
                if plp and not per_layer_params:
                    per_layer_params = plp
                tr_acc = t_info.get("train_acc")
                vl_acc = t_info.get("val_acc")
                if tr_acc is not None and vl_acc is not None and train_val_accs is None:
                    train_val_accs = (tr_acc, vl_acc)
            return (train_turn, seed, loaded_entries, device, None, train_val_accs,
                    total_train_steps, per_layer_steps, ensemble_steps, per_layer_params, 0.0, {})

        # ── Load train data from disk ──
        if train_datasets_list is not None:
            print(f"    [combined] Loading datasets from disk: {train_datasets_list}")
            train_df_full = load_data_combined(model_name, train_datasets_list, train_turn,
                                               columns=load_columns)
        else:
            train_df_full = load_data(model_name, train_dataset, train_turn,
                                      columns=load_columns)
            if train_dataset == test_dataset:
                train_df_full, _ = split_50_50(train_df_full, SPLIT_SEED)

        if balance and train_datasets_list is not None and '_source_dataset' in train_df_full.columns:
            seed_everywhere(seed)
            source_pids = {
                ds: set(train_df_full.loc[train_df_full['_source_dataset'] == ds, 'problem_id'].unique())
                for ds in train_datasets_list
            }
            n_per_source = {ds: len(pids) for ds, pids in source_pids.items()}
            min_n = min(n_per_source.values())
            balanced_frames = []
            for ds in train_datasets_list:
                ds_df = train_df_full[train_df_full['_source_dataset'] == ds].copy()
                ds_pids = np.array(sorted(source_pids[ds]))
                if len(ds_pids) > min_n:
                    sampled_pids = np.random.choice(ds_pids, size=min_n, replace=False)
                    ds_df = ds_df[ds_df['problem_id'].isin(sampled_pids)].copy()
                    print(f"    [balance] {ds}: subsampled {len(ds_pids)} → {min_n} problem_ids "
                          f"({len(ds_df)} rows)")
                else:
                    print(f"    [balance] {ds}: kept all {len(ds_pids)} problem_ids "
                          f"({len(ds_df)} rows)")
                balanced_frames.append(ds_df)
            train_df_full = pd.concat(balanced_frames, ignore_index=True)

        if '_source_dataset' in train_df_full.columns:
            train_df_full = train_df_full.drop(columns=['_source_dataset'])

        all_cols = embedding_col if isinstance(embedding_col, list) else [embedding_col]
        train_df = train_df_full

        if train_fraction < 1.0:
            seed_everywhere(seed)
            problem_ids = train_df['problem_id'].unique()
            n_problems_sample = max(1, int(round(train_fraction * len(problem_ids))))
            sampled_problem_ids = np.random.choice(problem_ids, size=n_problems_sample, replace=False)
            train_df = train_df[train_df['problem_id'].isin(sampled_problem_ids)].copy()
            print(f"    [Train Fraction] Subsampling {len(train_df)} entries via "
                  f"{n_problems_sample}/{len(problem_ids)} problem_ids (fraction={train_fraction})")

        train_df, val_df = split_val(train_df, seed, val_ratio=0.25)

        _wall_start = time.time()

        device = torch.device(f'cuda:0')
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

        embed_type = infer_embed_type(embedding_col)

        if embed_type in MULTI_TOKEN_TYPES:
            cols = all_cols
            print(f"    [multi-token '{embed_type}'] "
                  f"Combining {len(cols)} layer(s) into token sequence")

            X_train_mt, mask_train_mt = prepare_features_multi_token(train_df, cols, embed_type)
            X_val_mt,   mask_val_mt   = prepare_features_multi_token(val_df,   cols, embed_type)

            y_train_mt = train_df['is_correct'].values
            y_val_mt   = val_df['is_correct'].values

            final_model, _, total_train_steps = train_model_multi_token(
                X_train_mt, y_train_mt, X_val_mt, y_val_mt,
                seed, gpu_id=0,
                mask_train=mask_train_mt, mask_val=mask_val_mt,
                train_turn=train_turn, use_grid=use_grid,
            )
            profile_results = {}
            if profile:
                print(f"\n[PROFILE] Training forward pass — multi-token model (turn {train_turn}, seed {seed})")
                _pr = profile_model_pass(final_model, embedding_col, train_df, embed_type, device,
                                         tag=f"train_fwd turn{train_turn} seed{seed}")
                if _pr: profile_results['default'] = _pr
            model_entries = [(final_model, embedding_col, 'default', embedding_col)]

        elif model_type == 'nn' and isinstance(embedding_col, list):
            unique, counts = np.unique(train_df['is_correct'].values, return_counts=True)
            pos_weight = counts[0] / counts[1] if unique[1] == 1 else counts[1] / counts[0]
            pos_weight_tensor = torch.FloatTensor([pos_weight]).to(device)

            trained = []
            for col in embedding_col:
                print(f"    [nn multi-layer] Training MLP for layer col: {col}")
                X_train_l, y_train_l = prepare_features(train_df, col)
                X_val_l,   y_val_l   = prepare_features(val_df,   col)
                val_loss, model, steps = train_single_mlp(
                    X_train_l, y_train_l, X_val_l, y_val_l,
                    seed, device, pos_weight_tensor, use_grid=use_grid
                )
                total_train_steps    += steps
                per_layer_steps[col]  = steps
                per_layer_params[col] = sum(p.numel() for p in model.parameters())
                print(f"      -> val loss = {val_loss:.4f}, steps = {steps:,}, params = {per_layer_params[col]:,}")
                trained.append((col, val_loss, model))

            model_entries = []
            for col, val_loss_l, m in trained:
                layer_idx = col.split("_layer_")[1].split("_")[0]
                layer_label = f"layer_{layer_idx}"
                model_entries.append((m, col, layer_label, col))
                print(f"    [nn per-layer] layer_label={layer_label}, val_loss={val_loss_l:.4f}")

            best_col, best_val_loss, best_model = min(trained, key=lambda t: t[1])
            print(f"    [nn best-layer] Best: {best_col} (val loss={best_val_loss:.4f})")
            model_entries.append((best_model, best_col, 'best_layer', embedding_col))

            print(f"    [nn ensemble] Training ensemble weights on shared val set...")
            ensemble_model, ensemble_steps = train_ensemble_weights(trained, val_df, seed, device)
            model_entries.append((ensemble_model, embedding_col, 'ensemble', embedding_col))

            profile_results = {}
            if profile:
                print(f"\n[PROFILE] Training forward pass — best-layer MLP (turn {train_turn}, seed {seed})")
                _pr = profile_model_pass(best_model, best_col, train_df, embed_type, device,
                                         tag=f"train_fwd best_layer turn{train_turn} seed{seed}")
                if _pr: profile_results['best_layer'] = _pr
                if _pr: profile_results['ensemble'] = _pr  # same architecture

        else:
            X_train, y_train = prepare_features(train_df, embedding_col)
            X_val,   y_val   = prepare_features(val_df,   embedding_col)
            final_model, _, total_train_steps = train_model(
                X_train, y_train, X_val, y_val, seed,
                gpu_id=0, train_turn=train_turn, use_grid=use_grid
            )

            if not isinstance(embedding_col, list) and model_type == 'nn':
                final_model.eval()
                device_local = torch.device('cuda:0')
                X_tr, y_tr = prepare_features(train_df, embedding_col)
                X_vl, y_vl = prepare_features(val_df,   embedding_col)
                with torch.no_grad():
                    tr_preds = torch.sigmoid(final_model(torch.FloatTensor(X_tr).to(device_local))).cpu().numpy().flatten()
                    vl_preds = torch.sigmoid(final_model(torch.FloatTensor(X_vl).to(device_local))).cpu().numpy().flatten()
                train_acc = ((tr_preds >= 0.5).astype(int) == y_tr).mean()
                val_acc   = ((vl_preds >= 0.5).astype(int) == y_vl).mean()
                train_val_accs = (train_acc, val_acc)

            model_entries = [(final_model, embedding_col, 'default', embedding_col)]
            profile_results = {}
            if profile:
                print(f"\n[PROFILE] Training forward pass — {'MLP' if not isinstance(embedding_col, list) else 'Transformer'} (turn {train_turn}, seed {seed})")
                _pr = profile_model_pass(final_model, embedding_col, train_df, embed_type, device,
                                         tag=f"train_fwd turn{train_turn} seed{seed}")
                if _pr: profile_results['default'] = _pr

        save_dir = Path("./saved_models")
        save_dir.mkdir(exist_ok=True)
        frac_str  = "" if train_fraction == 1.0 else f"_frac{train_fraction}"
        same_split = (train_datasets_list is None) and (train_dataset == test_dataset)
        same_str   = "_same" if same_split else ""

        for model, eff_col, label, tag_col in model_entries:
            if isinstance(tag_col, list):
                layer_tag = "layers_" + "_".join(
                    c.split("_layer_")[1].split("_")[0] for c in tag_col
                ) + f"_{embed_type}"
            else:
                layer_tag = tag_col
            label_tag     = f"_{label}" if label != 'default' else ""
            save_filename = (
                f"model_{model_name}_{train_dataset}{same_str}_turn{train_turn}"
                f"_seed{seed}{frac_str}_{layer_tag}_{model_type}{label_tag}.pt"
            )
            model_meta = {
                "model_class": model.__class__.__name__,
                "init_args":   getattr(model, 'init_args', None),
                "architecture": {"input_dim": model.input_dim, "output_dim": model.output_dim},
                "training_info": {
                    "seed": seed, "train_turn": train_turn,
                    "train_dataset": train_dataset,
                    "train_datasets_list": train_datasets_list,
                    "test_dataset": test_dataset,
                    "embedding_col": embedding_col, "effective_embedding_col": eff_col,
                    "train_fraction": train_fraction, "model_type": model_type,
                    "label": label, "embed_type": embed_type,
                    "total_train_steps": total_train_steps,
                    "per_layer_steps":   per_layer_steps,
                    "per_layer_params":  per_layer_params,
                    "ensemble_steps":    ensemble_steps,
                    "use_grid": use_grid,
                    "balance": balance,
                }
            }
            if isinstance(model, EnsembleMLP):
                model_meta["training_info"]["ensemble_members"] = [
                    {"model_class": m.__class__.__name__,
                     "init_args":   getattr(m, 'init_args', None)}
                    for m in model.models
                ]
            if train_val_accs is not None:
                model_meta["training_info"]["train_acc"] = float(train_val_accs[0])
                model_meta["training_info"]["val_acc"]   = float(train_val_accs[1])
            torch.save({"model_state_dict": model.state_dict(), "model_meta": model_meta},
                       save_dir / save_filename)

        _wall_elapsed = time.time() - _wall_start
        print(f"  [timing] Turn {train_turn} Seed {seed}: training wall-clock = "
              f"{_wall_elapsed:.1f}s ({_wall_elapsed / 60:.2f} min), GPU {actual_gpu_id}")
        return (train_turn, seed,
                [(m, ec, lbl) for m, ec, lbl, _ in model_entries],
                device, None, train_val_accs, total_train_steps,
                per_layer_steps, ensemble_steps, per_layer_params, _wall_elapsed,
                profile_results)

    except Exception as e:
        import traceback
        _wall_elapsed = time.time() - _wall_start if _wall_start is not None else 0.0
        return (train_turn, seed, None, None, traceback.format_exc(), None, 0, {}, 0, {}, _wall_elapsed, {})

def process_test_eval(args_tuple):
    (model_name, train_turn, test_turn, seed,
     model, device, test_dataset, embedding_col, label, n_rollouts, profile,
     test_df_cached, frac) = args_tuple
    seed_everywhere(seed)
    try:
        all_cols = embedding_col if isinstance(embedding_col, list) else [embedding_col]
        test_df = test_df_cached.copy()

        if n_rollouts is not None:
            test_df = test_df[test_df['rollout_idx'] < n_rollouts].reset_index(drop=True)

        embed_type = infer_embed_type(embedding_col)

        model.eval()
        chunk_size = 1

        t_infer_start = time.time()
        if embed_type in MULTI_TOKEN_TYPES:
            predictions = _run_multi_token_inference(
                model, test_df, embedding_col, embed_type, device, chunk_size=chunk_size
            )
        elif isinstance(model, EnsembleMLP):
            all_preds = []
            n = len(test_df)
            with torch.no_grad():
                for start in range(0, n, chunk_size):
                    chunk = test_df.iloc[start:start + chunk_size]
                    x_dict = {
                        col: torch.FloatTensor(np.stack(chunk[col].values)).to(device)
                        for col in embedding_col
                    }
                    preds = model(x_dict).cpu().numpy().flatten()
                    all_preds.append(preds)
            predictions = np.concatenate(all_preds)
        else:
            X_test, _ = prepare_features(test_df, embedding_col)
            n = len(X_test)
            all_preds = []
            with torch.no_grad():
                for start in range(0, n, chunk_size):
                    chunk_X  = torch.FloatTensor(X_test[start:start + chunk_size]).to(device)
                    preds    = torch.sigmoid(model(chunk_X)).cpu().numpy().flatten()
                    all_preds.append(preds)
            predictions = np.concatenate(all_preds)
        infer_wall_s = time.time() - t_infer_start

        bon_acc = evaluate_best_of_n(test_df, predictions)
        wmv_acc = None
        if test_dataset not in ['kodcode', 'humaneval', 'bigcodebench_hard', 'ot_code']:
            wmv_acc = evaluate_weighted_majority(test_df, predictions)

        infer_profile = None
        if profile:
            embed_type_inf = infer_embed_type(embedding_col)
            infer_profile = profile_model_pass(model, embedding_col, test_df, embed_type_inf, device,
                               tag=f"inference {label} train{train_turn} test{test_turn} seed{seed}")

        test_acc = ((predictions >= 0.5).astype(int) == test_df['is_correct'].values).mean()

        return (train_turn, test_turn, seed, bon_acc, wmv_acc, label, None,
                test_acc, test_dataset, n_rollouts, frac, infer_wall_s, infer_profile)
    except Exception as e:
        import traceback
        return (train_turn, test_turn, seed, None, None, label, traceback.format_exc(),
                None, test_dataset, n_rollouts, frac, 0, None)

# ---------------------------------------------------------------------------
# Helper: build embedding_col and effective_model_type from a single
# (embed_type, layers_list, model_type_arg) triple.
# ---------------------------------------------------------------------------
def _build_embedding_col(embed_type, layers_list, model_type_arg):
    """Return (embedding_col, effective_model_type) for one embed_type/layers combo."""
    if len(layers_list) == 1:
        embedding_col = f"intermediate_layer_{layers_list[0]}_{embed_type}"
    else:
        embedding_col = [f"intermediate_layer_{l}_{embed_type}" for l in sorted(layers_list)]

    et = infer_embed_type(embedding_col)
    if et in MULTI_TOKEN_TYPES:
        effective_model_type = 'multi_token_transformer'
    else:
        effective_model_type = model_type_arg if isinstance(embedding_col, list) else 'nn'

    return embedding_col, effective_model_type


# ---------------------------------------------------------------------------
# run_one_embed_type: all training + eval for one (embed_type, layers) combo,
# reusing already-loaded data structures passed in from main().
# ---------------------------------------------------------------------------
def run_one_embed_type(
    *,
    embedding_col,
    effective_model_type,
    embed_type,
    args,
    train_dataset,
    train_datasets_list,
    train_turns,
    test_datasets_list,
    test_turns_dict,
    test_df_cache,
    train_fractions,
    seeds,
    gpu_ids,
    gpu_ids_str,
    rollout_counts,
    save_dir,
    is_same_split,
    mode_str,
    hpo_str,
    test_name,
    load_columns=None,
):
    """Train and evaluate for a single embed_type across all train_fractions."""

    print(f"\n{'=' * 70}")
    print(f"EMBED TYPE: {embed_type} | Layers: {embedding_col}")
    print(f"{'=' * 70}")

    def _expected_labels():
        if embed_type in MULTI_TOKEN_TYPES:
            return [('default', embedding_col)]
        elif effective_model_type == 'nn' and isinstance(embedding_col, list):
            per_layer = [
                (f"layer_{c.split('_layer_')[1].split('_')[0]}", c)
                for c in embedding_col
            ]
            return per_layer + [('best_layer', embedding_col), ('ensemble', embedding_col)]
        else:
            return [('default', embedding_col)]

    expected_labels = _expected_labels()

    def _all_checkpoints_exist(frac, train_turn, seed):
        for label, tag_col in expected_labels:
            fname = get_save_filename(
                args.model, train_dataset, train_turn, seed, frac,
                tag_col, effective_model_type, label, same_split=is_same_split
            )
            if not (save_dir / fname).exists():
                return False
        return True

    all_exist = bool(train_turns) and all(
        _all_checkpoints_exist(frac, tt, s)
        for frac in train_fractions for tt in train_turns for s in seeds
    )

    # ── Training ──────────────────────────────────────────────────────────
    # trained_models entries: (frac, train_turn, seed, model, device, eff_col, label)
    trained_models           = []
    train_val_accs_by_key    = {}  # (frac, train_turn, seed)
    total_train_steps_by_key = {}
    per_layer_steps_by_key   = {}
    ensemble_steps_by_key    = {}
    per_layer_params_by_key  = {}
    wall_elapsed_by_key      = {}
    profile_results_by_key   = {}

    if all_exist:
        print("All checkpoints found — skipping training, loading from disk.")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for frac in train_fractions:
            for train_turn in train_turns:
                for seed in seeds:
                    for label, tag_col in expected_labels:
                        fname = get_save_filename(
                            args.model, train_dataset, train_turn, seed, frac,
                            tag_col, effective_model_type, label, same_split=is_same_split
                        )
                        path = save_dir / fname
                        print(f"  Loading {path.name}")
                        ckpt = torch.load(path, map_location=device)
                        model, eff_col = load_model_from_checkpoint(path, device, ckpt=ckpt)
                        trained_models.append((frac, train_turn, seed, model, device, eff_col, label))
                        t_info = ckpt["model_meta"]["training_info"]
                        tr_acc = t_info.get("train_acc")
                        vl_acc = t_info.get("val_acc")
                        if tr_acc is not None and vl_acc is not None:
                            train_val_accs_by_key[(frac, train_turn, seed)] = (tr_acc, vl_acc)
                        steps = t_info.get("total_train_steps")
                        if steps is not None:
                            total_train_steps_by_key[(frac, train_turn, seed)] = steps
                        pls = t_info.get("per_layer_steps")
                        if pls:
                            per_layer_steps_by_key[(frac, train_turn, seed)] = pls
                        es = t_info.get("ensemble_steps")
                        if es is not None:
                            ensemble_steps_by_key[(frac, train_turn, seed)] = es
                        plp = t_info.get("per_layer_params")
                        if plp:
                            per_layer_params_by_key[(frac, train_turn, seed)] = plp
    else:
        train_tasks = []
        task_idx = 0
        for frac in train_fractions:
            for train_turn in train_turns:
                for seed in seeds:
                    gpu_idx = task_idx % args.num_gpus
                    train_tasks.append((
                        args.model, train_turn, seed, gpu_idx,
                        train_dataset, test_name, embedding_col,
                        gpu_ids_str, frac, effective_model_type,
                        args.profile, args.grid,
                        train_datasets_list,
                        args.balance,
                        load_columns,
                    ))
                    task_idx += 1

        train_wall_start = time.time()
        n_gpus_used      = args.num_gpus

        train_results = []
        if args.sequential:
            for task in train_tasks:
                result = process_train_seed(task)
                train_results.append(result)
        else:
            if not train_tasks:
                print("WARNING: No training tasks generated — nothing to train.")
            else:
                with mp.Pool(
                    processes=min(len(train_tasks), args.num_gpus * args.tasks_per_gpu),
                ) as pool:
                    train_results = pool.map(process_train_seed, train_tasks)

        train_wall_elapsed = time.time() - train_wall_start
        print(f"\n  [timing] Total training wall-clock time: "
              f"{train_wall_elapsed:.1f}s ({train_wall_elapsed / 60:.2f} min) "
              f"using {n_gpus_used} GPU(s)")

        # zip with tasks to recover frac (position 8 in each task tuple)
        for task, (train_turn, seed, model_entries, device, error,
                   train_val_accs, total_train_steps,
                   per_layer_steps, ensemble_steps, per_layer_params, wall_elapsed,
                   profile_results_task) in zip(train_tasks, train_results):
            frac = task[8]
            key  = (frac, train_turn, seed)
            if train_val_accs is not None:
                train_val_accs_by_key[key] = train_val_accs
            total_train_steps_by_key[key] = total_train_steps
            if per_layer_steps:
                per_layer_steps_by_key[key] = per_layer_steps
            if ensemble_steps:
                ensemble_steps_by_key[key] = ensemble_steps
            if per_layer_params:
                per_layer_params_by_key[key] = per_layer_params
            wall_elapsed_by_key[key] = wall_elapsed
            if profile_results_task:
                profile_results_by_key[key] = profile_results_task
            if error:
                print(f"Frac {frac} Train Turn {train_turn}, Seed {seed}: ERROR during training")
                print(f"  {error}")
            else:
                for model_inst, eff_col, label in model_entries:
                    trained_models.append((frac, train_turn, seed, model_inst, device, eff_col, label))

    # ── Evaluation ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    test_tasks = []
    for (frac, train_turn, seed, model_inst, device, eff_col, label) in trained_models:
        for test_ds in test_datasets_list:
            for test_turn in test_turns_dict[test_ds]:
                for nr in rollout_counts:
                    test_tasks.append((
                        args.model, train_turn, test_turn, seed,
                        model_inst, device, test_ds, eff_col, label, nr, args.profile,
                        test_df_cache[(test_ds, test_turn)],
                        frac,
                        ))
    results_list = []
    for task in test_tasks:
        seed_everywhere(task[3])  # seed is at position 3
        result = process_test_eval(task)
        results_list.append(result)

    # ── Collect results ───────────────────────────────────────────────────
    # Keys: (test_ds, label, nr, frac)
    bon_scores_by_key = defaultdict(list)
    wmv_scores_by_key = defaultdict(list)
    test_accs_by_key  = {}  # (test_ds, frac, train_turn, test_turn, seed, nr)
    infer_wall_by_key    = defaultdict(list)  # (test_ds, label, train_turn, test_turn, seed, nr, frac)
    infer_profile_by_key = {}  # same key

    for (train_turn, test_turn, seed, bon_acc, wmv_acc, label, error, test_acc, test_ds, nr, frac, infer_wall_s, infer_profile) in results_list:
        if error:
            print(f"[{test_ds}] Frac {frac} Train Turn {train_turn}, Test Turn {test_turn}, "
                  f"Seed {seed} [{label}] n_rollouts={nr}: Error")
            print(f"  {error}")
        else:
            bon_scores_by_key[(test_ds, label, nr, frac)].append(bon_acc)
            if wmv_acc is not None:
                wmv_scores_by_key[(test_ds, label, nr, frac)].append(wmv_acc)
            if test_acc is not None:
                test_accs_by_key[(test_ds, frac, train_turn, test_turn, seed, nr)] = test_acc
            infer_wall_by_key[(test_ds, label, nr, frac)].append(infer_wall_s)
            if infer_profile is not None:
                infer_profile_by_key[(test_ds, label, frac)] = infer_profile

    # ── Write results ─────────────────────────────────────────────────────
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = (
        results_dir /
        f"results_{args.model_type}_{args.model}_{train_dataset}_{test_name}_{embed_type}.txt"
    )

    using_combined  = train_datasets_list is not None
    using_dual_test = len(test_datasets_list) > 1

    if isinstance(embedding_col, list):
        layer_str = ','.join(c.split('_layer_')[1].split('_')[0] for c in embedding_col)
    else:
        layer_str = embedding_col.split('_layer_')[1].split('_')[0]
    args_str = f"--type {embed_type} --layer {layer_str} --model-type {args.model_type}"

    train_arg_str = (f"--train1 {args.train1} --train2 {args.train2}"
                     if using_combined else f"--train {train_dataset}")
    test_arg_str  = (f"--test1 {args.test1} --test2 {args.test2}"
                     if using_dual_test else f"--test {test_name}")

    with open(output_file, 'a') as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"Script: transformer_across_layers.py\n")
        f.write(f"Args: --model {args.model} {train_arg_str} {test_arg_str} "
                f"{args_str} --train-fractions {' '.join(str(fr) for fr in train_fractions)} "
                f"{'--rollouts' if args.rollouts else '--n-rollouts ' + str(args.n_rollouts)}"
                f"{' --grid' if args.grid else ''}"
                f"{' --balance' if args.balance else ''}\n")
        f.write(f"HPO: {hpo_str}\n")
        f.write(f"Mode: {mode_str}\n")
        f.write(f"Embedding: {embedding_col}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Train dataset(s): {train_datasets_list if using_combined else train_dataset}\n")
        f.write(f"Train (combined name): {train_dataset}\n")
        f.write(f"Balance sources: {args.balance}\n")
        f.write(f"Test dataset(s): {test_datasets_list}\n")
        f.write(f"Train Turns: {train_turns} | Test Turns: {test_turns_dict} | Seeds: {seeds}\n")
        f.write(f"Train data fractions: {train_fractions}\n")
        f.write(f"N rollouts (eval): {'4,8,12,all' if args.rollouts else args.n_rollouts}\n")
        f.write(f"GPUs used (--gpu-ids): {gpu_ids}\n\n")

        for frac in train_fractions:
            f.write(f"\n  === Train fraction: {frac} ===\n")
            for test_ds in test_datasets_list:
                f.write(f"\n  --- Test dataset: {test_ds} ---\n")
                for nr in rollout_counts:
                    nr_label = f"n_rollouts={nr}" if nr is not None else "n_rollouts=all"
                    f.write(f"  -- {nr_label} --\n")
                    for (ds_key, label, res_nr, res_frac), bon_scores in sorted(
                        bon_scores_by_key.items(),
                        key=lambda x: (x[0][0], x[0][1], x[0][2] if x[0][2] is not None else 16, x[0][3])
                    ):
                        if ds_key != test_ds or res_nr != nr or res_frac != frac:
                            continue
                        bon_mean = np.mean(bon_scores) if bon_scores else float('nan')
                        bon_std  = np.std(bon_scores)  if bon_scores else float('nan')
                        f.write(f"  [{label}] Best-of-N: {bon_mean:.4f} ± {bon_std:.4f}\n")
                        wmv_scores = wmv_scores_by_key.get((test_ds, label, nr, frac), [])
                        if wmv_scores:
                            f.write(f"  [{label}] Weighted Majority Vote: "
                                    f"{np.mean(wmv_scores):.4f} ± {np.std(wmv_scores):.4f}\n")

        if not isinstance(embedding_col, list) and effective_model_type == 'nn':
            f.write("\nPer-run classification accuracies (single-layer nn):\n")
            for (frac, tt, s), (tr_acc, vl_acc) in sorted(train_val_accs_by_key.items()):
                f.write(f"  Frac {frac} Train Turn {tt} Seed {s}: "
                        f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}\n")
            for (ds_key, frac, tt, tst, s, nr), t_acc in sorted(
                test_accs_by_key.items(),
                key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4],
                               x[0][5] if x[0][5] is not None else 16)
            ):
                f.write(f"  [{ds_key}] Frac {frac} Train Turn {tt} Test Turn {tst} "
                        f"Seed {s} n_rollouts={nr}: test_acc={t_acc:.4f}\n")
        f.write(f"{'=' * 70}\n")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"RESULTS — embed_type={embed_type}")
    print(f"{'=' * 70}")
    for frac in train_fractions:
        print(f"\n  === Train fraction: {frac} ===")
        for test_ds in test_datasets_list:
            print(f"\n  --- Test dataset: {test_ds} ---")
            for nr in rollout_counts:
                nr_label = f"n_rollouts={nr}" if nr is not None else "n_rollouts=all"
                print(f"  -- {nr_label} --")
                for (ds_key, label, res_nr, res_frac), bon_scores in sorted(
                    bon_scores_by_key.items(),
                    key=lambda x: (x[0][0], x[0][1], x[0][2] if x[0][2] is not None else 16, x[0][3])
                ):
                    if ds_key != test_ds or res_nr != nr or res_frac != frac:
                        continue
                    bon_mean = np.mean(bon_scores) if bon_scores else float('nan')
                    bon_std  = np.std(bon_scores)  if bon_scores else float('nan')
                    print(f"  [{label}] Best-of-N: {bon_mean:.4f} ± {bon_std:.4f}")
                    wmv_scores = wmv_scores_by_key.get((test_ds, label, nr, frac), [])
                    if wmv_scores:
                        print(f"  [{label}] Weighted Majority Vote: "
                              f"{np.mean(wmv_scores):.4f} ± {np.std(wmv_scores):.4f}")

    print(f"Results appended to {output_file}")

    # ── Write profiling file ──────────────────────────────────────────────────
    if args.profile and (profile_results_by_key or infer_wall_by_key):
        profiling_file = output_file.parent / (output_file.stem + "_profiling.txt")
        total_wall = sum(wall_elapsed_by_key.values())
        with open(profiling_file, 'a') as pf:
            pf.write(f"\n{'=' * 70}\n")
            pf.write(f"PROFILING — training forward pass\n")
            pf.write(f"{'=' * 70}\n")
            pf.write(f"  Total training wall-clock time: {total_wall:.1f}s "
                     f"({total_wall/60:.2f} min), {args.num_gpus} GPU(s)\n\n")

            for (frac, tt, s), pr in sorted(profile_results_by_key.items()):
                if abs(frac - 1.0) > 1e-9:
                    continue  # only write frac=1.0
                wall   = wall_elapsed_by_key.get((frac, tt, s), 0.0)
                steps  = total_train_steps_by_key.get((frac, tt, s), 0)
                for label, pdata in pr.items():
                    flops_raw  = pdata.get('flops_raw', 0)
                    macs_raw   = pdata.get('macs_raw', 0)
                    params_raw = pdata.get('params', '?')
                    batch_size = pdata.get('batch_size', 32)
                    latency_ms = pdata.get('latency_total_ms', '?')

                    flops_per_step = flops_raw  # already for full batch
                    train_flops_upper = flops_per_step * steps * 3
                    train_flops_lower = flops_per_step * steps * 2.5

                    pf.write(f"  [profile train | label={label}] "
                             f"model={args.model}, train={train_dataset}, "
                             f"turn={tt}, seed={s}\n")
                    pf.write(f"    FLOPs/sample : {flops_raw/batch_size:.4e}  "
                             f"(batch total: {flops_raw/1e6:.3f}M)\n")
                    pf.write(f"    MACs/sample  : {macs_raw/batch_size:.4e}  "
                             f"(batch total: {macs_raw/1e6:.3f}M)\n")
                    pf.write(f"    Params       : {params_raw}\n")
                    pf.write(f"    Latency/sample: {float(latency_ms)/batch_size*1000:.4f} ms  "
                             f"(batch: {latency_ms} ms, batch_size={batch_size})\n")
                    pf.write(f"    Training wall-clock time (excl. data loading): "
                             f"{wall:.1f}s ({wall/60:.2f} min), GPU(s): {gpu_ids}\n")
                    pf.write(f"    --- Training compute estimate (all trials included) ---\n")
                    pf.write(f"    Total gradient steps (all trials): {steps:,}\n")
                    if train_flops_upper > 0:
                        pf.write(f"    Training FLOPs estimate (×3 upper): {train_flops_upper:.4e}\n")
                        pf.write(f"    Training FLOPs estimate (×2.5 lower): {train_flops_lower:.4e}\n")
                    pf.write(f"    (per-step FLOPs = profiled fwd FLOPs for batch_size={batch_size})\n\n")

            pf.write(f"{'=' * 70}\n")
            # ── Inference section ──────────────────────────────────────────────
            if infer_wall_by_key:
                pf.write(f"\n{'=' * 70}\n")
                pf.write(f"PROFILING — inference forward pass (averaged over all turns and seeds)\n")
                pf.write(f"{'=' * 70}\n\n")

                nr_order = [2, 4, 8, 12, 16, None]

                # Collect all walls per (label, nr) across all test_ds, test_turns, seeds, frac=1.0
                label_nr_walls = defaultdict(lambda: defaultdict(list))
                for (test_ds_k, label_k, nr_k, frac_k), walls in infer_wall_by_key.items():
                    if abs(frac_k - 1.0) > 1e-9:
                        continue
                    label_nr_walls[label_k][nr_k].extend(walls)

                for label_k in sorted(label_nr_walls.keys()):
                    nr_dict = label_nr_walls[label_k]
                    pdata   = next(
                        (v for (ds, lbl, frac), v in infer_profile_by_key.items()
                         if lbl == label_k and abs(frac - 1.0) < 1e-9),
                        None
                    )

                    pf.write(f"  [profile infer | label={label_k}] model={args.model}\n")
                    pf.write(f"    Inference wall-clock averaged over all turns and seeds "
                             f"(excl. data loading):\n")
                    for nr_k in nr_order:
                        if nr_k not in nr_dict:
                            continue
                        walls    = nr_dict[nr_k]
                        mean_ms  = np.mean(walls) * 1000
                        std_ms   = np.std(walls)  * 1000
                        nr_label = str(nr_k) if nr_k is not None else "all"
                        pf.write(f"      n_rollouts={nr_label:<4}: "
                                 f"{mean_ms:8.2f} ms +/- {std_ms:6.2f} ms "
                                 f"(n={len(walls)})\n")
                    pf.write("\n")

                pf.write(f"{'=' * 70}\n")
        print(f"Profiling written to {profiling_file}")

    return output_file



def _load_test_worker(args_tuple):
    """Top-level worker for ProcessPoolExecutor: load one test pkl."""
    model_name, test_ds, test_turn, train_dataset, using_combined, parsed_data_dir, columns = args_tuple
    global PARSED_DATA_DIR
    PARSED_DATA_DIR = Path(parsed_data_dir)
    print(f"  Loading test {test_ds} turn{test_turn}...")
    df = load_data(model_name, test_ds, test_turn, columns=columns)
    if not using_combined and test_ds == train_dataset:
        _, df = split_50_50(df, SPLIT_SEED)
        print(f"    [train==test] Using held-out 50% of '{test_ds}' "
              f"({df['problem_id'].nunique()} problem_ids, {len(df)} rows)")
    return (test_ds, test_turn), df


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description='SCATR: train a classifier on LM intermediate-layer embeddings'
    )
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--train1', type=str, default=None)
    parser.add_argument('--train2', type=str, default=None)
    parser.add_argument('--test',  type=str, default=None)
    parser.add_argument('--test1', type=str, default=None)
    parser.add_argument('--test2', type=str, default=None)
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model-type', type=str, choices=['transformer', 'nn'], default='transformer')
    parser.add_argument('--layer', type=str, required=True,
                        help='Comma-separated layer indices.')
    parser.add_argument('--type', type=str, required=True,
                        choices=['mean', 'final', 'special', 'last_10', 'attn_weighted', 'all'],
                        help='Embedding type.')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--gpu-ids', type=str, default=None)
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--train-fractions', type=float, nargs='+', default=[1.0])
    parser.add_argument('--n-rollouts', type=int, default=None)
    parser.add_argument('--rollouts', action='store_true',
                        help='Evaluate at n_rollouts=4, 8, 12 and None (all) in one run.')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--tasks-per-gpu', type=int, default=1)
    args = parser.parse_args()

    # ── Validate rollout args ──────────────────────────────────────────────
    if args.rollouts and args.n_rollouts is not None:
        parser.error('--rollouts and --n-rollouts are mutually exclusive.')
    rollout_counts = [2, 4, 8, 12, None] if args.rollouts else [args.n_rollouts]

    # ── Validate train/test dataset args ──────────────────────────────────
    using_combined = args.train1 is not None or args.train2 is not None
    using_single   = args.train is not None
    if using_combined and using_single:
        parser.error("Cannot use --train together with --train1/--train2.")
    if using_combined and not (args.train1 and args.train2):
        parser.error("--train1 and --train2 must both be specified together.")
    if not using_combined and not using_single:
        parser.error("Must specify either --train or both --train1 and --train2.")

    using_dual_test  = args.test1 is not None or args.test2 is not None
    using_single_test = args.test is not None
    if using_dual_test and using_single_test:
        parser.error("Cannot use --test together with --test1/--test2.")
    if using_dual_test and not (args.test1 and args.test2):
        parser.error("--test1 and --test2 must both be specified together.")
    if not using_dual_test and not using_single_test:
        parser.error("Must specify either --test or both --test1 and --test2.")

    if using_dual_test:
        test_datasets_list = [args.test1, args.test2]
        test_name = f"{args.test1}_{args.test2}"
    else:
        test_datasets_list = [args.test]
        test_name = args.test

    if args.balance and not using_combined:
        parser.error("--balance requires --train1 and --train2.")

    # ── Dataset name setup ─────────────────────────────────────────────────
    if using_combined:
        train_dataset       = make_combined_train_name(args.train1, args.train2)
        train_datasets_list = [args.train1, args.train2]
        print(f"[Combined training] datasets: {train_datasets_list} → name: '{train_dataset}'")
        for _td in test_datasets_list:
            if _td in train_datasets_list:
                print(f"WARNING: test dataset '{_td}' is also in the combined training set.")
    else:
        train_dataset       = args.train
        train_datasets_list = None

    seed_everywhere(42)

    if not all(0 < f <= 1 for f in args.train_fractions):
        parser.error("--train-fractions values must all be in (0, 1].")

    # ── GPU setup ──────────────────────────────────────────────────────────
    if args.gpu_ids is None:
        default_num_gpus = 8
        args.num_gpus = default_num_gpus
        gpu_ids = list(range(default_num_gpus))
        args.gpu_ids = ','.join(str(g) for g in gpu_ids)
    else:
        gpu_ids = [int(i) for i in args.gpu_ids.split(',') if i.strip() != '']
        if len(gpu_ids) == 0:
            parser.error('No valid GPU ids specified in --gpu-ids')
        if args.num_gpus > len(gpu_ids):
            print(f"Warning: --num-gpus={args.num_gpus} > specified GPU ids={len(gpu_ids)}. "
                  f"Setting num_gpus to {len(gpu_ids)}.")
            args.num_gpus = len(gpu_ids)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpu_ids)
    gpu_ids_str = ','.join(str(i) for i in gpu_ids)

    seeds = [32, 42, 52]

    global PARSED_DATA_DIR
    PARSED_DATA_DIR = Path("./parsed_data")
    print(f"Using parsed data from {PARSED_DATA_DIR}")

    # ── Available turns ────────────────────────────────────────────────────
    if using_combined:
        turns1 = get_available_turns(args.model, args.train1)
        turns2 = get_available_turns(args.model, args.train2)
        train_turns = sorted(set(turns1) & set(turns2))
        if not train_turns:
            raise ValueError(
                f"No overlapping turns found between '{args.train1}' (turns {turns1}) "
                f"and '{args.train2}' (turns {turns2})."
            )
        print(f"[Combined] Available turns: {args.train1}={turns1}, "
              f"{args.train2}={turns2} → intersection={train_turns}")
    else:
        train_turns = get_available_turns(args.model, train_dataset)

    test_turns_dict = {ds: get_available_turns(args.model, ds) for ds in test_datasets_list}

    hpo_str = "random search (100 configs)" if args.grid else "Optuna TPE (96 trials)"
    save_dir = Path("./saved_models")

    is_same_split = (
        not using_combined
        and len(test_datasets_list) == 1
        and test_datasets_list[0] == train_dataset
    )

    # ── Pre-load test dataframes in parallel ──────────────────────────────
    from concurrent.futures import ProcessPoolExecutor, as_completed as pex_as_completed

    # Build embedding_col now so we know which columns to load.
    embed_type, layers_list = args.type, [int(x) for x in args.layer.split(',')]
    embedding_col, effective_model_type = _build_embedding_col(
        embed_type, layers_list, args.model_type
    )

    ec_cols = embedding_col if isinstance(embedding_col, list) else [embedding_col]
    load_columns = sorted(ec_cols)
    print(f"Column filter: loading only {len(load_columns)} embedding column(s): "
          f"{load_columns[:3]}{'...' if len(load_columns) > 3 else ''}")

    test_keys = [
        (test_ds, test_turn)
        for test_ds in test_datasets_list
        for test_turn in test_turns_dict[test_ds]
    ]
    test_tasks = [
        (args.model, test_ds, test_turn, train_dataset, using_combined,
         str(PARSED_DATA_DIR), load_columns)
        for test_ds, test_turn in test_keys
    ]
    print(f"Pre-loading {len(test_tasks)} test dataframe(s) in parallel...")
    test_df_cache = {}
    if test_tasks:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=len(test_tasks), mp_context=ctx) as ex:
            futs = [ex.submit(_load_test_worker, t) for t in test_tasks]
            for fut in pex_as_completed(futs):
                key, df = fut.result()
                test_df_cache[key] = df
    print(f"  Done. {len(test_df_cache)} test dataframe(s) cached.\n")

    if embed_type in MULTI_TOKEN_TYPES:
        n_layers = len(embedding_col) if isinstance(embedding_col, list) else 1
        mode_str = (
            f"MultiTokenTransformer ({embed_type}, {n_layers} layer(s), "
            f"tokens concatenated into sequence)"
        )
    elif isinstance(embedding_col, list):
        if effective_model_type == 'transformer':
            mode_str = f"Transformer ({len(embedding_col)} layers)"
        else:
            mode_str = f"Per-layer MLP + trained ensemble ({len(embedding_col)} layers)"
    else:
        mode_str = "MLP"

    print(f"\nModel: {args.model} | Train: {train_dataset} | "
          f"Test: {test_name} | Mode: {mode_str} | "
          f"Embedding: {embedding_col} | HPO: {hpo_str}")
    print(f"GPUs: {gpu_ids} | num_gpus: {args.num_gpus}")
    print(f"Train turns: {train_turns} | Test turns: {test_turns_dict}")

    out_file = run_one_embed_type(
        embedding_col=embedding_col,
        effective_model_type=effective_model_type,
        embed_type=embed_type,
        args=args,
        train_dataset=train_dataset,
        train_datasets_list=train_datasets_list,
        train_turns=train_turns,
        test_datasets_list=test_datasets_list,
        test_turns_dict=test_turns_dict,
        test_df_cache=test_df_cache,
        train_fractions=args.train_fractions,
        seeds=seeds,
        gpu_ids=gpu_ids,
        gpu_ids_str=gpu_ids_str,
        rollout_counts=rollout_counts,
        save_dir=save_dir,
        is_same_split=is_same_split,
        mode_str=mode_str,
        hpo_str=hpo_str,
        test_name=test_name,
        load_columns=load_columns,
    )
    print(f"\nAll experiments done. Results written to: {out_file}")


if __name__ == "__main__":
    main()