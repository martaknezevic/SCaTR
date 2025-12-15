"""
Transformer-based method for learning from logprobs.

Based on optimize_metric_transformer.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys

from methods.base_method import BaseMethod
from methods.method_registry import register_method


class TransformerMetricModel(nn.Module):
    """Transformer encoder that consumes per-token top-logprobs and outputs a scalar score."""

    def __init__(self, num_top_logprobs: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, mlp_hidden: int = 64, dropout: float = 0.1, max_len=2048, use_projection: bool = True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_projection = use_projection
        if use_projection:
            self.input_proj = nn.Linear(num_top_logprobs, d_model)
            self.input_dim = d_model
        else:
            self.input_proj = None
            self.input_dim = num_top_logprobs
        self.max_len = max_len

        # Build transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                     dim_feedforward=d_model*4, dropout=dropout,
                                     batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_dim))
        
        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(self.max_len + 1, self.input_dim)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
        
        self.dropout = nn.Dropout(dropout)

        # Storage for attention weights
        self.attention_weights = []

        # Patch attention modules
        self._patch_attention_modules()

    def _patch_attention_modules(self):
        """Patch self-attention forward methods to capture attention weights."""
        for layer_idx, layer in enumerate(self.encoder_layers):
            original_forward = layer.self_attn.forward

            def make_patched_forward(orig_forward):
                def patched_forward(query, key, value, key_padding_mask=None,
                                  need_weights=True, attn_mask=None,
                                  average_attn_weights=True, **kwargs):
                    attn_output, attn_weights = orig_forward(
                        query, key, value,
                        key_padding_mask=key_padding_mask,
                        need_weights=need_weights,
                        attn_mask=attn_mask,
                        average_attn_weights=average_attn_weights,
                        **kwargs
                    )

                    if self.training and attn_weights is not None:
                        cls_attn = attn_weights[:, 0, :]
                        self.attention_weights.append(cls_attn)

                    return attn_output, attn_weights

                return patched_forward

            layer.self_attn.forward = make_patched_forward(original_forward)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, num_top_logprobs)
        # mask: (batch, seq_len) - True for padding positions
        self.attention_weights = []

        batch_size = x.size(0)
        if self.use_projection:
            x = self.input_proj(x)  # -> (batch, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Adjust mask for CLS token if provided
        if mask is not None:
            # Prepend False for CLS token (always attend to it)
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len+1)
        
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, seq_len+1)
        pos_emb = self.pos_embedding(positions)                           # (1, seq_len+1, d_model)
        x = x + pos_emb

        # Pass through transformer layers with mask
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=mask)

        cls_emb = x[:, 0, :]
        score = self.mlp(cls_emb).squeeze(-1)

        return score

    def get_attention_l1_loss(self):
        """Compute L1 regularization on CLS token attention weights."""
        if not self.attention_weights:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        all_attentions = []
        for attn in self.attention_weights:
            attn_to_tokens = attn[:, 1:]
            all_attentions.append(attn_to_tokens)

        if not all_attentions:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        stacked = torch.stack(all_attentions, dim=0)
        l1_loss = torch.mean(torch.abs(stacked))

        return l1_loss

@register_method('transformer')
class TransformerMethod(BaseMethod):
    """
    Transformer-based method for learning confidence scores from logprobs.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Required parameters (from method.config in Hydra)
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.mlp_hidden = config['mlp_hidden']
        self.dropout = config['dropout']
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.batch_problems = config['batch_problems']
        self.margin = config['margin']
        self.attention_l1_weight = config['attention_l1_weight']
        self.patience = config['patience']
        self.early_stop_metric = config['early_stop_metric']
        self.use_projection = config['use_projection']
        
        # Optional parameters
        self.tail_tokens = config.get('tail_tokens', None)
        self.seed = config.get('seed', None)
        self.loss_type = config.get('loss_type', 'hinge')  # 'hinge' or 'bce'
        self.use_weighted_loss = config.get('use_weighted_loss', False)
        
        self.num_top_logprobs = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Train transformer model.

        Args:
            train_data: Training data
            val_data: Validation data for early stopping

        Returns:
            Training metrics
        """
        # Set seed for reproducibility
        if self.seed is not None:
            self.set_seed(self.seed)
            print(f"Set random seed to {self.seed} for weight initialization")
        
        print(f"\n{'='*80}")
        print(f"{self.name}: Training transformer model...")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Organize by problem
        train_problems = defaultdict(list)
        for item in train_data:
            train_problems[item['problem_id']].append(item)

        val_problems = defaultdict(list)
        if val_data:
            for item in val_data:
                val_problems[item['problem_id']].append(item)

        # Determine num_top_logprobs
        self.num_top_logprobs = train_data[0]['logprobs'].shape[1] if train_data else 10
        print(f"Number of top logprobs per token: {self.num_top_logprobs}")

        # Create model
        self.model = TransformerMetricModel(
            num_top_logprobs=self.num_top_logprobs,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
            max_len=self.tail_tokens,
            use_projection=self.use_projection
        )
        self.model.to(self.device)

        print(f"Model architecture:")
        print(f"  d_model: {self.d_model}, nhead: {self.nhead}, layers: {self.num_layers}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        sys.stdout.flush()

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_metric = -1e9
        best_epoch = -1
        epochs_no_improve = 0        
        best_state = None        
        history = {'train_loss': [], 'train_attn_l1': [], 'train_acc': [], 'val_acc': [], 'val_loss': []}

        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_attn_l1 = 0.0

            problem_ids = sorted(list(train_problems.keys()))
            import random
            random.shuffle(problem_ids)


            batch_loss = 0.0
            batch_attn_l1 = 0.0
            batch_count = 0
            optimizer.zero_grad()

            for i, pid in enumerate(problem_ids):
                rollouts = train_problems[pid]

                # Build batch
                tensors = []
                truths = []
                for r in rollouts:
                    lp = r['logprobs']
                    if self.tail_tokens is not None and self.tail_tokens > 0:
                        lp = lp[-self.tail_tokens:]
                    tensors.append(torch.tensor(lp, dtype=torch.float32))
                    truths.append(1 if r['is_correct'] else 0)

                if not tensors:
                    continue

                # Determine max_len: use tail_tokens if specified, otherwise max in batch
                if self.tail_tokens is not None and self.tail_tokens > 0:
                    max_len = self.tail_tokens
                else:
                    max_len = max(t.shape[0] for t in tensors)
                
                batch = torch.zeros((len(tensors), max_len, tensors[0].shape[1]), dtype=torch.float32)
                mask = torch.ones((len(tensors), max_len), dtype=torch.bool)  # True = padding
                
                for bi, t in enumerate(tensors):
                    # Pad or truncate to max_len
                    seq_len = min(t.shape[0], max_len)
                    batch[bi, :seq_len, :] = t[:seq_len]
                    mask[bi, :seq_len] = False  # False = real token

                batch = batch.to(self.device)
                mask = mask.to(self.device)
                scores = self.model(batch, mask=mask)

                truths = torch.tensor(truths, dtype=torch.bool, device=self.device)
                pos_idx = torch.nonzero(truths).squeeze(-1)
                neg_idx = torch.nonzero(~truths).squeeze(-1)

                if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                    continue

                # Compute loss based on loss_type
                if self.loss_type == 'bce':
                    # Binary cross-entropy loss
                    targets = truths.float()
                    probs = torch.sigmoid(scores)
                    
                    if self.use_weighted_loss:
                        # Weight by inverse frequency
                        n_pos = pos_idx.numel()
                        n_neg = neg_idx.numel()
                        total = n_pos + n_neg
                        pos_weight = total / (2.0 * n_pos)
                        neg_weight = total / (2.0 * n_neg)
                        weights = torch.where(truths, pos_weight, neg_weight)
                        task_loss = F.binary_cross_entropy(probs, targets, weight=weights, reduction='sum')
                    else:
                        task_loss = F.binary_cross_entropy(probs, targets, reduction='sum')
                else:
                    # Hinge loss
                    s_pos = scores[pos_idx].unsqueeze(1)
                    s_neg = scores[neg_idx].unsqueeze(0)
                    diffs = s_pos - s_neg
                    losses = F.relu(self.margin - diffs)
                    
                    if self.use_weighted_loss:
                        # Weight by inverse frequency
                        n_pos = pos_idx.numel()
                        n_neg = neg_idx.numel()
                        total = n_pos + n_neg
                        pos_weight = total / (2.0 * n_pos)
                        neg_weight = total / (2.0 * n_neg)
                        # Apply weights to loss matrix
                        weight_matrix = pos_weight * neg_weight
                        task_loss = (losses * weight_matrix).sum()
                    else:
                        task_loss = losses.sum()

                # Add attention sparsity regularization
                attention_l1_loss = self.model.get_attention_l1_loss()
                loss = task_loss + self.attention_l1_weight * attention_l1_loss

                loss.backward()
                batch_loss += loss.item()
                batch_attn_l1 += attention_l1_loss.item()
                batch_count += 1

                if batch_count >= self.batch_problems:
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += batch_loss
                    epoch_attn_l1 += batch_attn_l1
                    batch_loss = 0.0
                    batch_attn_l1 = 0.0
                    batch_count = 0

            # Final optimizer step
            if batch_count > 0:
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += batch_loss
                epoch_attn_l1 += batch_attn_l1

            # Evaluate on training and validation sets
            train_acc = self._evaluate_accuracy(train_problems)
            val_acc = self._evaluate_accuracy(val_problems if val_problems else train_problems)
            val_loss = self._evaluate_loss(val_problems if val_problems else train_problems)

            history['train_loss'].append(epoch_loss)
            history['train_attn_l1'].append(epoch_attn_l1)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch}/{self.num_epochs} - train_loss={epoch_loss:.4f} "
                  f"train_attn_l1={epoch_attn_l1:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
            sys.stdout.flush()
            
            # Log to wandb if enabled
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': epoch_loss,
                        'train_attn_l1': epoch_attn_l1,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
            except ImportError:
                pass

            # Early stopping
            if self.early_stop_metric == 'val_loss':
                current_metric = -val_loss
            else:
                current_metric = val_acc

            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {self.patience} epochs)")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\n✓ Restored model to best checkpoint at epoch {best_epoch}")
            sys.stdout.flush()
        else:
            print(f"\n⚠ Warning: No best model state saved, using final model")
            sys.stdout.flush()

        self.is_trained = True

        return {
            'best_epoch': best_epoch,
            'best_val_metric': best_val_metric if self.early_stop_metric == 'val_acc' else -best_val_metric,
            'final_epoch': epoch,
            'history': history
        }

    def _evaluate_accuracy(self, problems: Dict[str, List[Dict]]) -> float:
        """Evaluate accuracy on problems."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for problem_id, rollouts in problems.items():
                tensors = []
                truths = []
                for r in rollouts:
                    lp = r['logprobs']
                    if self.tail_tokens is not None and self.tail_tokens > 0:
                        lp = lp[-self.tail_tokens:]
                    tensors.append(torch.tensor(lp, dtype=torch.float32))
                    truths.append(r['is_correct'])

                if not tensors:
                    continue

                # Determine max_len: use tail_tokens if specified, otherwise max in batch
                if self.tail_tokens is not None and self.tail_tokens > 0:
                    max_len = self.tail_tokens
                else:
                    max_len = max(t.shape[0] for t in tensors)
                
                batch = torch.zeros((len(tensors), max_len, tensors[0].shape[1]), dtype=torch.float32)
                mask = torch.ones((len(tensors), max_len), dtype=torch.bool)
                
                for i, t in enumerate(tensors):
                    seq_len = min(t.shape[0], max_len)
                    batch[i, :seq_len, :] = t[:seq_len]
                    mask[i, :seq_len] = False

                batch = batch.to(self.device)
                mask = mask.to(self.device)
                scores = self.model(batch, mask=mask)
                best_idx = int(torch.argmax(scores).cpu().numpy())
                if truths[best_idx]:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0

    def _evaluate_loss(self, problems: Dict[str, List[Dict]]) -> float:
        """Evaluate loss on problems."""
        self.model.eval()
        total_loss = 0.0
        total_problems = 0

        with torch.no_grad():
            for problem_id, rollouts in problems.items():
                tensors = []
                truths = []
                for r in rollouts:
                    lp = r['logprobs']
                    if self.tail_tokens is not None and self.tail_tokens > 0:
                        lp = lp[-self.tail_tokens:]
                    tensors.append(torch.tensor(lp, dtype=torch.float32))
                    truths.append(1 if r['is_correct'] else 0)

                if not tensors:
                    continue

                # Determine max_len: use tail_tokens if specified, otherwise max in batch
                if self.tail_tokens is not None and self.tail_tokens > 0:
                    max_len = self.tail_tokens
                else:
                    max_len = max(t.shape[0] for t in tensors)
                
                batch = torch.zeros((len(tensors), max_len, tensors[0].shape[1]), dtype=torch.float32)
                mask = torch.ones((len(tensors), max_len), dtype=torch.bool)
                
                for i, t in enumerate(tensors):
                    seq_len = min(t.shape[0], max_len)
                    batch[i, :seq_len, :] = t[:seq_len]
                    mask[i, :seq_len] = False

                batch = batch.to(self.device)
                mask = mask.to(self.device)
                scores = self.model(batch, mask=mask)
                truths_t = torch.tensor(truths, dtype=torch.bool, device=self.device)
                pos_idx = torch.nonzero(truths_t).squeeze(-1)
                neg_idx = torch.nonzero(~truths_t).squeeze(-1)

                if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                    continue

                # Compute loss based on loss_type
                if self.loss_type == 'bce':
                    targets = truths_t.float()
                    probs = torch.sigmoid(scores)
                    
                    if self.use_weighted_loss:
                        n_pos = pos_idx.numel()
                        n_neg = neg_idx.numel()
                        total = n_pos + n_neg
                        pos_weight = total / (2.0 * n_pos)
                        neg_weight = total / (2.0 * n_neg)
                        weights = torch.where(truths_t, pos_weight, neg_weight)
                        total_loss += F.binary_cross_entropy(probs, targets, weight=weights, reduction='sum').item()
                    else:
                        total_loss += F.binary_cross_entropy(probs, targets, reduction='sum').item()
                else:
                    # Hinge loss
                    s_pos = scores[pos_idx].unsqueeze(1)
                    s_neg = scores[neg_idx].unsqueeze(0)
                    diffs = s_pos - s_neg
                    losses = F.relu(self.margin - diffs)
                    
                    if self.use_weighted_loss:
                        n_pos = pos_idx.numel()
                        n_neg = neg_idx.numel()
                        total = n_pos + n_neg
                        pos_weight = total / (2.0 * n_pos)
                        neg_weight = total / (2.0 * n_neg)
                        weight_matrix = pos_weight * neg_weight
                        total_loss += (losses * weight_matrix).sum().item()
                    else:
                        total_loss += losses.sum().item()
                
                total_problems += 1

        return total_loss / total_problems if total_problems > 0 else 0.0

    def predict_confidence(self, data_item: Dict) -> float:
        """
        Predict confidence using trained transformer.

        Args:
            data_item: Dictionary with 'logprobs' key

        Returns:
            Confidence score
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Method must be trained before prediction")

        self.model.eval()

        lp = data_item['logprobs']
        if self.tail_tokens is not None and self.tail_tokens > 0:
            lp = lp[-self.tail_tokens:]
            max_len = self.tail_tokens
        else:
            max_len = len(lp)

        # Create padded batch with mask
        tensor = torch.tensor(lp, dtype=torch.float32)
        batch = torch.zeros((1, max_len, tensor.shape[1]), dtype=torch.float32)
        mask = torch.ones((1, max_len), dtype=torch.bool)
        
        # Fill in actual sequence
        seq_len = min(tensor.shape[0], max_len)
        batch[0, :seq_len, :] = tensor[:seq_len]
        mask[0, :seq_len] = False  # False = real tokens
        
        batch = batch.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            score = self.model(batch, mask=mask).cpu().item()

        return score

    def save(self, output_dir: str) -> Optional[str]:
        """Save trained model (framework interface)."""
        if self.model is None:
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / 'transformer_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_top_logprobs': self.num_top_logprobs,
            'config': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'mlp_hidden': self.mlp_hidden,
                'dropout': self.dropout,
                'num_epochs': self.num_epochs,
                'lr': self.lr,
                'batch_problems': self.batch_problems,
                'margin': self.margin,
                'attention_l1_weight': self.attention_l1_weight,
                'patience': self.patience,
                'early_stop_metric': self.early_stop_metric,
                'tail_tokens': self.tail_tokens,
                'seed': self.seed,
                'loss_type': self.loss_type,
                'use_weighted_loss': self.use_weighted_loss,
                'use_projection': self.use_projection
            }
        }, model_path)
        
        return str(model_path)
    
    def load(self, model_path: str) -> None:
        """Load trained model (framework interface)."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        config = checkpoint['config']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.mlp_hidden = config['mlp_hidden']
        self.dropout = config['dropout']
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.batch_problems = config['batch_problems']
        self.margin = config['margin']
        self.attention_l1_weight = config['attention_l1_weight']
        self.patience = config['patience']
        self.early_stop_metric = config['early_stop_metric']
        self.tail_tokens = config['tail_tokens']
        self.seed = config.get('seed')
        self.loss_type = config.get('loss_type', 'hinge')
        self.use_weighted_loss = config.get('use_weighted_loss', False)
        self.use_projection = config.get('use_projection', True)
        
        # Load model
        self.num_top_logprobs = checkpoint['num_top_logprobs']
        self.model = TransformerMetricModel(
            num_top_logprobs=self.num_top_logprobs,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
            max_len=self.tail_tokens,
            use_projection=self.use_projection
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.is_trained = True

    def save(self, save_dir: Path) -> None:
        """Save model and metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_top_logprobs': self.num_top_logprobs
        }, save_dir / 'model.pt')

        # Save metadata
        metadata = {
            'name': self.name,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'mlp_hidden': self.mlp_hidden,
            'dropout': self.dropout,
            'num_epochs': self.num_epochs,
            'lr': self.lr,
            'batch_problems': self.batch_problems,
            'margin': self.margin,
            'attention_l1_weight': self.attention_l1_weight,
            'patience': self.patience,
            'early_stop_metric': self.early_stop_metric,
            'tail_tokens': self.tail_tokens,
            'seed': self.seed,
            'loss_type': self.loss_type,
            'use_weighted_loss': self.use_weighted_loss,
            'is_trained': self.is_trained
        }
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ {self.name} saved to {save_dir}")

    def load(self, load_dir: Path) -> None:
        """Load model and metadata."""
        # Load metadata
        with open(load_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.d_model = metadata['d_model']
        self.nhead = metadata['nhead']
        self.num_layers = metadata['num_layers']
        self.mlp_hidden = metadata['mlp_hidden']
        self.dropout = metadata['dropout']
        self.num_epochs = metadata['num_epochs']
        self.lr = metadata['lr']
        self.batch_problems = metadata['batch_problems']
        self.margin = metadata['margin']
        self.attention_l1_weight = metadata['attention_l1_weight']
        self.patience = metadata['patience']
        self.early_stop_metric = metadata['early_stop_metric']
        self.tail_tokens = metadata['tail_tokens']
        self.seed = metadata.get('seed')
        self.loss_type = metadata.get('loss_type', 'hinge')
        self.use_weighted_loss = metadata.get('use_weighted_loss', False)
        self.is_trained = metadata['is_trained']

        # Load model
        checkpoint = torch.load(load_dir / 'model.pt', map_location=self.device)
        self.num_top_logprobs = checkpoint['num_top_logprobs']

        self.model = TransformerMetricModel(
            num_top_logprobs=self.num_top_logprobs,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        print(f"\n✓ {self.name} loaded from {load_dir}")
