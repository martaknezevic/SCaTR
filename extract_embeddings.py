"""
Add intermediate-layer hidden-state embeddings to parsed response data.

Each input .pkl file must be a DataFrame containing at minimum:
  - problem       : the user prompt / problem statement
  - response_text : the model's generated response

For each row, the chat template is applied to the problem to produce a prompt
string, which is concatenated with the response text and tokenized as a single
sequence.  The combined sequence is passed through the model in one forward
pass, and hidden states are extracted from the specified transformer layers and
pooled in one or more ways.

For every (layer, embedding_type) pair requested, a column named
  intermediate_layer_{i}_{type}
is added to the DataFrame and written back to the same .pkl file.

Embedding types
---------------
final        : hidden state of the final (last non-padding) token.
               Output shape per example: (hidden_dim,)
mean         : attention-mask-weighted mean pool over all tokens.
               Output shape per example: (hidden_dim,)
special      : hidden states of code-landmark tokens in the response — specifically
               `return` keywords and function-definition colons (`def ...:` endings),
               stacked into a matrix.  Falls back to the final token when no
               landmarks are found.
               Output shape per example: (n_landmarks, hidden_dim)  [variable length]
last_10      : hidden states of the last 10 non-padding tokens, left-padded with
               the first token's hidden state when the sequence is shorter than 10.
               Output shape per example: (10, hidden_dim)
attn_weighted: for each attention head at the requested layer, the weighted sum of
               all token hidden states using the attention weights directed at the
               final token.  Output shape per example: (n_heads, hidden_dim)
all          : hidden states of every non-padding token.
               Output shape per example: (n_tokens, hidden_dim)  [variable length]

Multi-GPU parallelism
---------------------
Rows are distributed across GPUs using Python multiprocessing (one process per GPU).
Each worker loads its own copy of the model and processes its assigned row slice
independently. Results are concatenated in the original row order before saving.

Usage example
-------------
  python extract_embeddings.py \\
      --model Qwen/Qwen3-1.7B \\
      --dataset humaneval \\
      --layers 12,20,28 \\
      --embedding-types final,mean \\
      --num-gpus 4
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import sys
import multiprocessing as mp
import re

# Embedding types supported
ALL_EMBEDDING_TYPES = ['final', 'mean', 'special', 'last_10', 'attn_weighted', 'all']


class EmbeddingExtractor:
    """
    Load a decoder-only LLM and extract hidden-state embeddings from its
    intermediate transformer layers.

    Uses output_hidden_states=True so that all requested layer activations are
    returned in a single forward pass.  Supports all standard HuggingFace
    AutoModel / AutoModelForCausalLM architectures.
    """

    def __init__(self, model_name, device=None, gpu_id=None):
        if device:
            self.device = device
        elif gpu_id is not None:
            self.device = f'cuda:{gpu_id}'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.is_gptoss = "gpt-oss" in model_name.lower() or "gptoss" in model_name.lower()

        print(f"[GPU {gpu_id if gpu_id is not None else 'auto'}] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if 'cuda' in self.device:
            torch_dtype = "auto"
        else:
            torch_dtype = torch.float32

        if self.is_gptoss:
            # GPT-OSS model config has "kernels-community/vllm-flash-attn3" baked in
            # and passes that key to ALL_ATTENTION_FUNCTIONS at runtime. Register it
            # as an alias for flash_attention_2 so the runtime lookup succeeds.
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            if "kernels-community/vllm-flash-attn3" not in ALL_ATTENTION_FUNCTIONS:
                ALL_ATTENTION_FUNCTIONS["kernels-community/vllm-flash-attn3"] = \
                    ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        model_kwargs = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map={"": self.device},
        )

        try:
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.model.to(self.device)
        self.model.eval()

        if self.is_gptoss:
            # The GPT-OSS config hard-codes the attention implementation key to
            # "kernels-community/vllm-flash-attn3", which must also be set on the
            # loaded model's config objects so that internal attention dispatch resolves
            # to the correct implementation after loading.
            object.__setattr__(self.model.config, "_attn_implementation_internal", "kernels-community/vllm-flash-attn3")
            if hasattr(self.model, "model"):
                object.__setattr__(self.model.model.config, "_attn_implementation_internal", "kernels-community/vllm-flash-attn3")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder models: left-pad so the final token of each sequence aligns
        # at the rightmost position.
        self.tokenizer.padding_side = 'left'

        print(f"[GPU {gpu_id if gpu_id is not None else 'auto'}] ✓ Model loaded "
              f"(hidden_size: {self.model.config.hidden_size})")

    def _capture_layers_standard(self, inputs, layer_indices, need_attentions=False,
                                  embedding_types=None):
        """
        Run a single forward pass with output_hidden_states=True and return the
        hidden states (and optionally attention weights) for the requested layers.

        HuggingFace convention: hidden_states[0] is the token-embedding output;
        hidden_states[i+1] is the output of transformer layer i.

        When the only requested embedding type is 'final', the last token is sliced
        immediately to avoid keeping full (batch, seq_len, hidden_dim) tensors in
        memory longer than necessary.  With left-padding the final non-pad token
        is always at the last position.

        Returns:
            captured_hidden : dict mapping layer_idx -> tensor
            captured_attn   : dict mapping layer_idx -> attention tensor (empty if
                              need_attentions=False)
        """
        if embedding_types is None:
            embedding_types = []
        final_only = set(embedding_types) <= {"final"}

        with torch.no_grad():
            out = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=need_attentions,
            )

        captured_hidden = {}
        captured_attn = {}

        for idx in layer_indices:
            # hidden_states[0] = embedding layer, hidden_states[idx+1] = after layer idx
            hs = out.hidden_states[idx + 1].detach()
            if final_only:
                captured_hidden[idx] = hs[:, -1, :]
            else:
                captured_hidden[idx] = hs

            if need_attentions and out.attentions is not None:
                captured_attn[idx] = out.attentions[idx].detach()

        del out
        return captured_hidden, captured_attn

    # ------------------------------------------------------------------
    # Pooling helpers
    # ------------------------------------------------------------------

    def mean_pool(self, hidden_states, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        masked_hidden = hidden_states * mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        return sum_hidden / sum_mask.clamp(min=1e-9)

    def extract_final_token(self, hidden_states, attention_mask):
        # With left-padding, the final non-pad token is always at the last position.
        return hidden_states[:, -1, :]

    # ------------------------------------------------------------------
    # Special (landmark) token helpers
    # ------------------------------------------------------------------

    def find_landmark_indices_from_ids(self, input_ids, response_start_idx):
        """
        Find token indices of code landmarks within the response portion of a
        (possibly left-padded) token ID sequence.

        Landmarks are:
          - `return` keywords
          - The closing colon of `def` function signatures (i.e., the `:` that ends
            `def foo(...):`)

        Only tokens at or after response_start_idx are examined, so that prompt and
        padding tokens are excluded.

        Args:
            input_ids          : list of int token IDs for the padded sequence as it
                                 was fed to the model (pad tokens prepended).
            response_start_idx : token index where the response begins, already
                                 adjusted for left-padding.

        Returns:
            List of token indices into input_ids corresponding to landmark positions.
            Empty list if none are found.
        """
        landmark_indices = []
        for tok_idx in range(response_start_idx, len(input_ids)):
            tok_str = self.tokenizer.decode([input_ids[tok_idx]])
            if 'return' in tok_str:
                landmark_indices.append(tok_idx)
            elif ':' in tok_str:
                preceding = self.tokenizer.decode(
                    input_ids[max(response_start_idx, tok_idx - 20):tok_idx]
                )
                if re.search(r'\bdef\b', preceding):
                    landmark_indices.append(tok_idx)
        return landmark_indices

    def extract_special_tokens(self, hidden_states_single, landmark_indices, fallback_idx):
        """
        Gather hidden states at landmark token positions for a single sequence.

        If landmark_indices is non-empty, returns the stacked hidden states at those
        positions — shape (n_landmarks, hidden_dim).  If no landmarks were found,
        falls back to the final non-padding token — shape (1, hidden_dim).
        """
        if landmark_indices:
            indices = torch.tensor(landmark_indices, device=hidden_states_single.device)
            return hidden_states_single[indices]
        else:
            return hidden_states_single[fallback_idx].unsqueeze(0)

    # ------------------------------------------------------------------
    # Main extraction entry point
    # ------------------------------------------------------------------

    def extract_embeddings_batch(self, texts, layer_indices, prompt_texts=None,
                                  batch_size=1, gpu_id=None, embedding_types=None):
        """
        Extract embeddings for a list of prompt+response texts across the requested
        layers and types.

        Each text is a prompt (from apply_chat_template) concatenated with a
        response string, tokenized as a single sequence.  Sequences are processed
        in mini-batches of `batch_size`; the tokenizer handles left-padding within
        each mini-batch.  A single forward pass captures all requested layer
        activations, which are then pooled into each requested embedding type.
        Layer tensors are freed immediately after pooling to keep peak GPU memory
        low.

        Args:
            texts           : list of str — full prompt+response strings.
            layer_indices   : list of transformer layer indices to extract from.
            prompt_texts    : list of str (one per text) — just the prompt portion.
                              Required for the 'special' embedding type; used to
                              locate where the response begins in the tokenized
                              sequence.
            batch_size      : number of sequences per forward pass (default 1).
            gpu_id          : GPU index used for tqdm position labelling.
            embedding_types : list of embedding type strings (see module docstring).

        Returns:
            dict mapping 'layer_{i}_{type}' -> numpy array (or list for variable-length
            types 'special' and 'all').
        """
        if embedding_types is None:
            embedding_types = ['final', 'mean']

        compute_special = 'special' in embedding_types
        compute_last_10 = 'last_10' in embedding_types
        compute_attn_weighted = 'attn_weighted' in embedding_types
        compute_all = 'all' in embedding_types

        if compute_special and prompt_texts is None:
            raise ValueError("prompt_texts must be provided when embedding_types includes 'special'")

        embeddings_dict = {}
        for layer_idx in layer_indices:
            for et in embedding_types:
                embeddings_dict[f'layer_{layer_idx}_{et}'] = []

        landmark_counts = [] if compute_special else None
        gpu_prefix = f"[GPU {gpu_id}]" if gpu_id is not None else ""

        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"{gpu_prefix} Extracting embeddings",
                      leave=False,
                      position=gpu_id if gpu_id is not None else 0):

            batch_texts = texts[i:i + batch_size]
            batch_prompt_texts = prompt_texts[i:i + batch_size] if prompt_texts is not None else None

            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                add_special_tokens=False,  # chat template already includes specials
            ).to(self.device)

            padded_seq_len = inputs['input_ids'].size(1)

            batch_landmark_indices = []
            if compute_special:
                for b in range(len(batch_texts)):
                    prompt_ids = self.tokenizer(
                        batch_prompt_texts[b], add_special_tokens=False
                    ).input_ids
                    seq_len_unpadded = int(inputs['attention_mask'][b].sum())
                    pad_offset = padded_seq_len - seq_len_unpadded
                    response_start_idx = pad_offset + len(prompt_ids)
                    fallback_idx = padded_seq_len - 1  # last position under left-padding

                    input_ids_list = inputs['input_ids'][b].tolist()
                    lm_indices = self.find_landmark_indices_from_ids(
                        input_ids=input_ids_list,
                        response_start_idx=response_start_idx,
                    )
                    batch_landmark_indices.append((lm_indices, fallback_idx))
                    landmark_counts.append(len(lm_indices) if lm_indices else 0)

            captured_hidden, captured_attn = self._capture_layers_standard(
                inputs, layer_indices, need_attentions=compute_attn_weighted,
                embedding_types=embedding_types,
            )

            for layer_idx in layer_indices:
                hidden_state = captured_hidden[layer_idx]  # (batch, seq_len, hidden_dim) or (batch, hidden_dim) if final_only

                if 'final' in embedding_types:
                    final_only = set(embedding_types) <= {"final"}
                    if final_only:
                        final_embeds = hidden_state  # already (batch, hidden)
                    else:
                        final_embeds = self.extract_final_token(hidden_state, inputs['attention_mask'])
                    embeddings_dict[f'layer_{layer_idx}_final'].append(
                        final_embeds.float().cpu().numpy())

                if 'mean' in embedding_types:
                    mean_embeds = self.mean_pool(hidden_state, inputs['attention_mask'])
                    embeddings_dict[f'layer_{layer_idx}_mean'].append(
                        mean_embeds.float().cpu().numpy())

                if compute_special:
                    for b, (lm_indices, fallback_idx) in enumerate(batch_landmark_indices):
                        special_emb = self.extract_special_tokens(
                            hidden_states_single=hidden_state[b],
                            landmark_indices=lm_indices,
                            fallback_idx=fallback_idx,
                        )
                        embeddings_dict[f'layer_{layer_idx}_special'].append(
                            special_emb.float().cpu().numpy())

                if compute_last_10:
                    for b in range(len(batch_texts)):
                        seq_len = int(inputs['attention_mask'][b].sum())
                        # With left-padding, valid tokens are at positions [end - seq_len : end]
                        end = hidden_state.size(1)
                        start = end - min(seq_len, 10)
                        last10 = hidden_state[b, start:end]
                        if last10.shape[0] < 10:
                            first_valid_idx = end - seq_len
                            pad = hidden_state[b, first_valid_idx:first_valid_idx + 1].expand(10 - last10.shape[0], -1)
                            last10 = torch.cat([pad, last10], dim=0)
                        embeddings_dict[f'layer_{layer_idx}_last_10'].append(
                            last10.float().cpu().numpy())

                if compute_attn_weighted:
                    attn_layer = captured_attn.get(layer_idx)
                    if attn_layer is None:
                        raise RuntimeError(
                            f"Attention weights not captured for layer {layer_idx}. "
                            "The model may not expose attention in its layer output tuple."
                        )
                    for b in range(len(batch_texts)):
                        # With left-padding, the final token is at the last position.
                        final_tok_idx = hidden_state.size(1) - 1
                        weights = attn_layer[b, :, final_tok_idx, :]
                        hidden = hidden_state[b]
                        attn_emb = weights @ hidden
                        embeddings_dict[f'layer_{layer_idx}_attn_weighted'].append(
                            attn_emb.float().cpu().numpy())

                if compute_all:
                    for b in range(len(batch_texts)):
                        seq_len = int(inputs['attention_mask'][b].sum())
                        # With left-padding, valid tokens are at the end of the sequence.
                        end = hidden_state.size(1)
                        start = end - seq_len
                        all_emb = hidden_state[b, start:end]
                        embeddings_dict[f'layer_{layer_idx}_all'].append(
                            all_emb.float().cpu().numpy())

                # Free this layer's tensor immediately once extraction is done
                del captured_hidden[layer_idx]
                if layer_idx in captured_attn:
                    del captured_attn[layer_idx]

            del inputs, captured_hidden, captured_attn
            torch.cuda.empty_cache()

        for key in embeddings_dict:
            if key.endswith('_special') or key.endswith('_all'):
                pass  # variable-length per-example — leave as list
            elif key.endswith('_last_10') or key.endswith('_attn_weighted'):
                embeddings_dict[key] = np.stack(embeddings_dict[key], axis=0)
            else:
                embeddings_dict[key] = np.vstack(embeddings_dict[key])

        if compute_special and landmark_counts:
            print(f"{gpu_prefix} Avg landmarks per response: {np.mean(landmark_counts):.2f}")

        return embeddings_dict


def process_file_rows(filepath, model_name, layer_indices, batch_size=1,
                      gpu_id=None, row_start=None, row_end=None,
                      df_subset=None, system_prompt=None,
                      embedding_types=None):
    """
    Process a contiguous slice of rows from one .pkl file and return their embeddings.

    This is the per-GPU worker's main computation function.  It loads the model once,
    then for each row applies the chat template to the problem to produce a prompt
    string and concatenates the response text to form the full input.  Tokenization
    of the combined string happens inside extract_embeddings_batch, which processes
    rows in chunks of 1000 to keep CPU memory bounded.

    Returns a dict mapping column name -> array/list, covering exactly the rows in
    df_subset (row_start to row_end).
    """
    if embedding_types is None:
        embedding_types = ['final', 'mean']

    gpu_prefix = f"[GPU {gpu_id}]" if gpu_id is not None else ""
    print(f"{gpu_prefix} Processing rows {row_start}-{row_end} ({len(df_subset)} rows)")

    extractor = EmbeddingExtractor(model_name, gpu_id=gpu_id)

    full_texts = []
    prompt_texts = []

    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset),
                         desc=f"{gpu_prefix} Building texts", leave=False,
                         position=gpu_id if gpu_id is not None else 0):
        problem = "" if pd.isna(row['problem']) or row['problem'] is None else str(row['problem'])
        response = "" if pd.isna(row['response_text']) or row['response_text'] is None else str(row['response_text'])

        prompt_text = extractor.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_texts.append(prompt_text + response)
        prompt_texts.append(prompt_text)

    chunk_size = 1000  # process rows in chunks to bound CPU-side memory for large files
    all_embeddings_dict = {}
    for layer_idx in layer_indices:
        for et in embedding_types:
            all_embeddings_dict[f'layer_{layer_idx}_{et}'] = []

    for chunk_start in tqdm(range(0, len(full_texts), chunk_size),
                            desc=f"{gpu_prefix} Processing chunks",
                            leave=False,
                            position=gpu_id if gpu_id is not None else 0):
        chunk_end = min(chunk_start + chunk_size, len(full_texts))
        chunk_texts = full_texts[chunk_start:chunk_end]
        chunk_prompt_texts = prompt_texts[chunk_start:chunk_end]

        chunk_embeddings = extractor.extract_embeddings_batch(
            chunk_texts,
            layer_indices=layer_indices,
            prompt_texts=chunk_prompt_texts,
            batch_size=batch_size,
            gpu_id=gpu_id,
            embedding_types=embedding_types,
        )

        for key in chunk_embeddings:
            all_embeddings_dict[key].append(chunk_embeddings[key])

        del chunk_embeddings
        torch.cuda.empty_cache()

    for key in all_embeddings_dict:
        if key.endswith('_special') or key.endswith('_all'):
            all_embeddings_dict[key] = [arr for chunk in all_embeddings_dict[key] for arr in chunk]
        elif key.endswith('_last_10') or key.endswith('_attn_weighted'):
            all_embeddings_dict[key] = np.concatenate(all_embeddings_dict[key], axis=0)
        else:
            all_embeddings_dict[key] = np.vstack(all_embeddings_dict[key])

    result = {}
    for layer_idx in layer_indices:
        for et in embedding_types:
            key = f'layer_{layer_idx}_{et}'
            col_name = f'intermediate_layer_{layer_idx}_{et}'
            result[col_name] = all_embeddings_dict[key]

    del extractor, all_embeddings_dict, full_texts, prompt_texts
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return result


def process_rows_worker(args):
    """
    Multiprocessing entry point: unpack args, configure the GPU environment, and
    call process_file_rows.  Returns a tuple of (gpu_id, row_start, row_end,
    embeddings_dict, error_traceback) so the parent process can reassemble results
    in order and surface any per-worker failures.

    CUDA_VISIBLE_DEVICES is set before any CUDA code runs so that each subprocess
    sees only its assigned physical GPU as device 0.
    """
    (filepath, model_name, layer_indices, batch_size,
     gpu_id, row_start, row_end, df_subset, system_prompt,
     embedding_types) = args

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    is_gptoss = "gpt-oss" in model_name.lower() or "gptoss" in model_name.lower()
    if is_gptoss:
        # Each worker is a fresh subprocess, so the attention-function alias registered
        # in EmbeddingExtractor.__init__ is not inherited.  Pre-register it here before
        # any transformers code is imported so the lookup succeeds when the model loads.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if "kernels-community/vllm-flash-attn3" not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS["kernels-community/vllm-flash-attn3"] = \
                ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    try:
        embeddings_dict = process_file_rows(
            filepath=filepath,
            model_name=model_name,
            layer_indices=layer_indices,
            batch_size=batch_size,
            gpu_id=0,
            row_start=row_start,
            row_end=row_end,
            df_subset=df_subset,
            system_prompt=system_prompt,
            embedding_types=embedding_types,
        )
        return (gpu_id, row_start, row_end, embeddings_dict, None)
    except Exception as e:
        import traceback
        return (gpu_id, row_start, row_end, None, traceback.format_exc())


def process_file(filepath, model_name, layer_indices, batch_size=1,
                 overwrite=False, gpu_id=None, gpu_ids=None, embedding_types=None):
    """
    Orchestrate multi-GPU embedding extraction for a single .pkl file.

    Reads the system prompt from the dataset-specific config YAML, splits the
    DataFrame rows evenly across the provided GPU IDs, spawns one worker process
    per GPU, then collects and merges results before writing back to the original
    file atomically (via a .tmp.pkl rename).

    Skips the file without modification if the requested embedding columns already
    exist, unless --overwrite is set.
    """
    if embedding_types is None:
        embedding_types = ['final', 'mean']
    if gpu_ids is None:
        gpu_ids = [0] if gpu_id is None else [gpu_id]

    gpu_prefix = f"[GPUs {gpu_ids}]"

    print(f"\n{'='*70}")
    print(f"{gpu_prefix} Processing: {filepath.name}")
    print(f"{'='*70}")

    df = pd.read_pickle(filepath)

    existing_cols = []
    for layer_idx in layer_indices:
        for et in embedding_types:
            col_name = f'intermediate_layer_{layer_idx}_{et}'
            if col_name in df.columns:
                existing_cols.append(col_name)

    if existing_cols and not overwrite:
        print(f"{gpu_prefix}  Embedding columns already exist: {existing_cols}")
        print(f"{gpu_prefix}    Use --overwrite to replace")
        return

    if 'problem' not in df.columns or 'response_text' not in df.columns:
        print(f"{gpu_prefix} Missing required columns (problem, response_text)")
        return

    import yaml
    dataset_guess = filepath.parent.name
    if dataset_guess == "math500":
        dataset_guess = "math"

    config_path = Path(f"./config_{dataset_guess}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found for dataset '{dataset_guess}'")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        system_prompt = config.get("generation", {}).get("system_prompt")
        if not system_prompt:
            raise KeyError
    except Exception:
        raise KeyError(f"Could not find system_prompt in config_{dataset_guess}.yaml")
    if not isinstance(system_prompt, str):
        raise ValueError("Invalid system_prompt format in config file")

    n_rows = len(df)
    n_gpus = len(gpu_ids)
    rows_per_gpu = n_rows // n_gpus

    print(f"{gpu_prefix} Splitting {n_rows} rows across {n_gpus} GPUs (~{rows_per_gpu} rows each)")

    tasks = []
    for i, gpu_id in enumerate(gpu_ids):
        row_start = i * rows_per_gpu
        row_end = n_rows if i == n_gpus - 1 else (i + 1) * rows_per_gpu
        df_subset = df.iloc[row_start:row_end].copy()
        tasks.append((
            filepath, model_name, layer_indices, batch_size,
            gpu_id, row_start, row_end, df_subset, system_prompt,
            embedding_types,
        ))
        print(f"{gpu_prefix}   GPU {gpu_id}: rows {row_start}-{row_end} ({len(df_subset)} rows)")

    print(f"{gpu_prefix} Starting parallel processing with {n_gpus} workers...")

    if n_gpus == 1:
        results = [process_rows_worker(tasks[0])]
    else:
        with mp.Pool(processes=n_gpus) as pool:
            results = pool.map(process_rows_worker, tasks)

    print(f"{gpu_prefix} Combining results from all GPUs...")

    errors = [(gpu_id, err) for gpu_id, _, _, _, err in results if err is not None]
    if errors:
        print(f"{gpu_prefix} ERRORS occurred:")
        for gpu_id, err in errors:
            print(f"  GPU {gpu_id}: {err}...")
        return

    results = sorted(results, key=lambda x: x[1])

    new_cols = {}
    for layer_idx in layer_indices:
        for et in embedding_types:
            col_name = f'intermediate_layer_{layer_idx}_{et}'
            chunks = [embeddings_dict[col_name]
                      for _, _, _, embeddings_dict, _ in results]
            if et == 'special' or et == 'all':
                combined = [arr for chunk in chunks for arr in chunk]
            else:
                combined = list(np.concatenate(chunks, axis=0))
            new_cols[col_name] = combined
            print(f"{gpu_prefix}   Added {col_name}")
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    tmp = filepath.with_suffix(".tmp.pkl")
    df.to_pickle(tmp)
    tmp.rename(filepath)
    print(f"{gpu_prefix} ✓ Saved {len(df)} rows with "
          f"{len(layer_indices) * len(embedding_types)} embedding columns to {filepath.name}")


def get_layer_range(model_name):
    """
    Utility: load the model config and print the valid range of layer indices.

    Useful for determining which values to pass to --layers without running a
    full embedding job.  Tries AutoModel first, falls back to AutoModelForCausalLM
    for models that require a language-model head to load.
    """
    try:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, trust_remote_code=True)
    n_layers = None
    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(model.config, attr):
            n_layers = int(getattr(model.config, attr))
            break
    if n_layers is None:
        raise RuntimeError(f"Could not determine layer count for model {model_name}")
    print(f"\nValid values for layer index i (option --layers) are integers from 0 to {n_layers-1} (inclusive).")
    return n_layers


def main():
    parser = argparse.ArgumentParser(description='Add intermediate layer embeddings to parsed data files')
    parser.add_argument('--model', type=str, default='allenai/OLMo-2-1124-7B-Instruct')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--turn', type=int, default=None)
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--gpu-ids', type=str, default=None)
    parser.add_argument('--layers', type=str, default='')
    parser.add_argument('--embedding-types', type=str, default='final,mean',
                        help=f'Comma-separated embedding types to compute. '
                             f'Options: {ALL_EMBEDDING_TYPES}. Default: final,mean. '
                             f'Example: --embedding-types final,mean,special')
    args = parser.parse_args()

    embedding_types = [t.strip() for t in args.embedding_types.split(',')]
    invalid = [t for t in embedding_types if t not in ALL_EMBEDDING_TYPES]
    if invalid:
        parser.error(f"Unknown embedding types: {invalid}. Valid options: {ALL_EMBEDDING_TYPES}")

    model_dir_name = None
    if args.model == "allenai/OLMo-2-1124-7B-Instruct":
        model_dir_name = "ollmo7b"
    elif args.model == "Qwen/Qwen3-1.7B":
        model_dir_name = "qwen1_7b"
    elif args.model == "openai/gpt-oss-20b":
        model_dir_name = "gptoss"
    elif args.model == "Qwen/Qwen3-30B-A3B":
        model_dir_name = "qwen30b"
    elif args.model == "Qwen/Qwen2.5-14B-Instruct":
        model_dir_name = "qwen14b"

    global PARSED_DATA_DIR
    PARSED_DATA_DIR = Path(f"./parsed_data/{model_dir_name}/{args.dataset}/")

    if args.layers.strip():
        layer_indices = [int(x) for x in args.layers.split(',')]
    else:
        layer_indices = []
    if not layer_indices:
        parser.error("Must specify --layers")

    print("="*70)
    print("ADD INTERMEDIATE LAYER EMBEDDINGS")
    print("="*70)
    print(f"Model: {args.model} | Batch: {args.batch_size}")
    print(f"Intermediate Layers: {layer_indices} | Types: {embedding_types}")
    print(f"Total columns to add: {len(layer_indices) * len(embedding_types)}")

    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.num_gpus))

    print(f"Using GPUs: {gpu_ids} (row-level parallelism)")

    if not PARSED_DATA_DIR.exists():
        print(f"Directory not found: {PARSED_DATA_DIR}")
        sys.exit(1)

    pkl_files = sorted(PARSED_DATA_DIR.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {PARSED_DATA_DIR}")
        sys.exit(1)

    if args.turn is not None:
        pkl_files = [f for f in pkl_files
                     if (m := re.match(r'^turn(\d+)\.pkl$', f.name)) and int(m.group(1)) == args.turn]
        if not pkl_files:
            print("No matching files found")
            sys.exit(1)

    filtered_pkls = []
    for f in pkl_files:
        filtered_pkls.append(f)
    pkl_files = filtered_pkls

    print(f"\nProcessing {len(pkl_files)} files")

    for filepath in pkl_files:
        try:
            df_test_exist = pd.read_pickle(filepath)
            existing_cols = [f"intermediate_layer_{li}_{et}"
                             for li in layer_indices for et in embedding_types
                             if f"intermediate_layer_{li}_{et}" in df_test_exist.columns]
            if existing_cols and not args.overwrite:
                print(f"Skipping {filepath.name}: columns already exist: {existing_cols}")
                continue
        except Exception as e:
            print(f"Warning: could not check columns in {filepath.name}: {e}")

        try:
            process_file(
                filepath,
                model_name=args.model,
                layer_indices=layer_indices,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                gpu_ids=gpu_ids,
                embedding_types=embedding_types,
            )
        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
            print("Aborting.")
            sys.exit(1)


if __name__ == "__main__":
    main()