"""
Attention-based token selection method for learning from logprobs.

Extracts attention patterns from a 2-layer transformer, computes attention statistics,
and uses learned importance scores to select the most relevant tokens for prediction.
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
import time

try:
    from entmax import entmax15
except ImportError:
    print("Warning: entmax not installed. Install with: pip install entmax")
    # Fallback to softmax
    def entmax15(logits, dim=-1):
        return F.softmax(logits, dim=dim)

from methods.base_method import BaseMethod
from methods.method_registry import register_method


class SimpleAttentionStats(nn.Module):
    """
    Simplified attention statistics for shallow (2-layer) transformers.
    No rollout needed - just smart aggregation of layer-specific patterns.
    """
    def __init__(self, num_layers=2, use_cls=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_cls = use_cls
        
        # Learnable layer weights - let model decide which layer matters more
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
    def forward(self, attentions, attention_mask=None):
        """
        attentions: list of 2 tensors [B, H, T, T] where T includes CLS token at position 0
        Returns: [B, T, stats_dim] for all tokens including CLS
        """
        B, H, T, _ = attentions[0].shape
        
        # Process each layer separately
        layer_stats = []
        
        for layer_idx, attn in enumerate(attentions):
            # Mean over heads: [B, T, T]
            attn_mean = attn.mean(dim=1)
            
            # Add residual connection (important even for 2 layers!)
            eye = torch.eye(T, device=attn.device).unsqueeze(0)
            attn_mean = 0.5 * attn_mean + 0.5 * eye
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
            
            # Basic statistics per layer
            received = attn_mean.sum(dim=1)  # [B, T] - how much attention each position receives
            given = attn_mean.sum(dim=2)     # [B, T] - how much attention each position gives
            self_attn = torch.diagonal(attn_mean, dim1=1, dim2=2)  # [B, T] - self-attention
            
            layer_stat = torch.stack([received, given, self_attn], dim=-1)
            layer_stats.append(layer_stat)
        
        # Stack layers: [B, T, num_layers, 3]
        layer_stats = torch.stack(layer_stats, dim=2)
        
        # Weighted combination using learned layer weights
        layer_weights_norm = F.softmax(self.layer_weights, dim=0)
        weighted_stats = (layer_stats * layer_weights_norm.view(1, 1, -1, 1)).sum(dim=2)
        # Result: [B, T, 3]
        
        stats_list = [weighted_stats]
        
        # Add per-layer stats (let importance network see both layers)
        # Flatten num_layers and stats dimensions: [B, T, num_layers, 3] -> [B, T, num_layers*3]
        per_layer_stats = layer_stats.reshape(B, T, -1)
        stats_list.append(per_layer_stats)
        
        if self.use_cls:
            # CLS statistics from both layers
            # For each token, compute how much it interacts with CLS
            cls_stats = []
            for attn in attentions:
                attn_mean = attn.mean(dim=1)  # [B, T, T] where position 0 is CLS
                # Attention FROM CLS TO each position
                cls_to_all = attn_mean[:, 0, :]  # [B, T]
                # Attention FROM each position TO CLS
                all_to_cls = attn_mean[:, :, 0]  # [B, T]
                cls_stats.extend([cls_to_all, all_to_cls])
            
            stats_list.extend(cls_stats)
        
        # Concatenate all statistics
        stats = torch.cat(stats_list, dim=-1)
        
        # Mask padding
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            stats = stats * mask
        
        return stats


class LightweightAttentionStats(nn.Module):
    """
    Even simpler: just use last layer + basic stats.
    Best for very shallow models or when training from scratch.
    """
    def __init__(self, use_cls=True):
        super().__init__()
        self.use_cls = use_cls
        
    def forward(self, attentions, attention_mask=None):
        """
        Only uses the LAST layer attention (most task-relevant).
        """
        # Just use last layer
        attn = attentions[-1]  # [B, H, T, T]
        
        # Mean over heads
        attn_mean = attn.mean(dim=1)  # [B, T, T]
        
        # Add residual
        B, T, _ = attn_mean.shape
        eye = torch.eye(T, device=attn.device).unsqueeze(0)
        attn_mean = 0.5 * attn_mean + 0.5 * eye
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
        
        # Compute statistics
        received = attn_mean.sum(dim=1)   # [B, T]
        given = attn_mean.sum(dim=2)      # [B, T]
        self_attn = torch.diagonal(attn_mean, dim1=1, dim2=2)  # [B, T]
        max_received, _ = attn_mean.max(dim=1)  # [B, T]
        
        stats_list = [received, given, self_attn, max_received]
        
        if self.use_cls:
            cls_attn = attn_mean[:, 0, :]
            to_cls_attn = attn_mean[:, :, 0]
            stats_list.extend([cls_attn, to_cls_attn])
        
        stats = torch.stack(stats_list, dim=-1)
        
        # Mask padding
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            stats = stats * mask
        
        return stats


class TokenImportance(nn.Module):
    """
    Simpler importance network for shallow models.
    """
    def __init__(self, hidden_dim, attn_dim, dropout=0.1):
        super().__init__()
        # Single layer is enough for shallow models
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + attn_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hidden_states, attn_stats, attention_mask=None):
        x = torch.cat([hidden_states, attn_stats], dim=-1)
        logits = self.net(x).squeeze(-1)
        
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask == 0, -1e9)
        
        return logits


class TokenSelector(nn.Module):
    """Token selector using entmax for sparse selection."""
    def __init__(self, k=None, temperature=1.0):
        super().__init__()
        self.k = k
        self.temperature = temperature

    def forward(self, logits, attention_mask=None):
        logits = logits / self.temperature
        weights = entmax15(logits, dim=-1)
        
        if self.k is not None:
            topk_values, topk_indices = torch.topk(weights, k=self.k, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, topk_indices, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        num_selected = (weights > 1e-6).sum(dim=-1)
        return weights, num_selected


class TokenAggregator(nn.Module):
    """Aggregate tokens using weighted sum."""
    def forward(self, hidden_states, weights):
        weights = weights.unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1)
        return pooled


class MetricHead(nn.Module):
    """Simpler metric head for shallow models."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, pooled):
        return self.net(pooled).squeeze(-1)


class AttentionSelectionModel(nn.Module):
    """
    Optimized for shallow (2-layer) transformers trained from scratch.
    
    Key differences:
    - No rollout (overkill for 2 layers)
    - Learnable layer weights (model decides which layer matters)
    - Simpler networks (less parameters to train from scratch)
    - Option to use only last layer
    """
    def __init__(self, num_top_logprobs: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, max_len: int = 2048,
                 k: int = 10, temperature: float = 1.0, use_both_layers: bool = True,
                 use_projection: bool = True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_projection = use_projection
        self.use_both_layers = use_both_layers
        
        # Input projection
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
        
        self.dropout = nn.Dropout(dropout)
        
        # Token selection modules (simple selector just stores k)
        self.selector = TokenSelector(k=k, temperature=temperature)
        self.aggregator = TokenAggregator()
        self.metric_head = MetricHead(d_model, dropout=dropout)
        
        # Storage for attention weights
        self.attention_weights = []

    def forward(self, x, mask=None, return_details=False):
        """
        Args:
            x: (batch, seq_len, num_top_logprobs)
            mask: (batch, seq_len) - True for padding positions
            return_details: Whether to return detailed outputs
        
        Returns:
            score or dict with detailed outputs
        """
        self.attention_weights = []
        
        batch_size = x.size(0)
        if self.use_projection:
            x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Adjust mask for CLS token if provided
        if mask is not None:
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Add positional embeddings
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        # Pass through transformer layers and collect attentions manually
        attentions = []
        for layer_idx, layer in enumerate(self.encoder_layers):
            # Extract attention weights manually before layer forward
            attn_weights = self._compute_layer_attention(x, mask, layer)
            attentions.append(attn_weights)
            
            # Forward pass through the layer
            x = layer(x, src_key_padding_mask=mask)
        
        # attentions is now a list of attention tensors [B, H, T+1, T+1] (includes CLS at position 0)
        hidden = x  # [B, T+1, d_model]
        
        # Extract CLS attention from last layer (most task-relevant)
        # Use average over heads
        last_attn = attentions[-1].mean(dim=1)  # [B, T+1, T+1]
        
        # Get attention FROM CLS to all tokens
        cls_attention = last_attn[:, 0, :]  # [B, T+1]
        
        # Mask out padding tokens
        if mask is not None:
            cls_attention = cls_attention.masked_fill(mask, -1e9)
        
        # Select top-k tokens based on CLS attention
        k = min(self.selector.k if self.selector.k else hidden.shape[1], hidden.shape[1])
        topk_values, topk_indices = torch.topk(cls_attention, k=k, dim=-1)  # [B, k]
        
        # Create sparse weights (only top-k are non-zero)
        weights = torch.zeros_like(cls_attention)  # [B, T+1]
        weights.scatter_(1, topk_indices, torch.softmax(topk_values, dim=-1))
        
        # Weighted aggregation of selected tokens
        pooled = self.aggregator(hidden, weights)
        
        # Predict metric
        metric = self.metric_head(pooled)
        
        if return_details:
            num_selected = (weights > 1e-6).sum(dim=-1)
            return {
                "metric": metric,
                "token_weights": weights,
                "cls_attention": cls_attention,
                "num_selected": num_selected,
                "topk_indices": topk_indices
            }
        else:
            return metric
    
    def _compute_layer_attention(self, x, mask, layer):
        """
        Manually compute attention weights for a transformer layer.
        
        Args:
            x: input tensor [B, T, D]
            mask: padding mask [B, T] (True for padding)
            layer: TransformerEncoderLayer
            
        Returns:
            attention weights [B, H, T, T]
        """
        B, T, D = x.shape
        
        # Access the self-attention module
        attn_module = layer.self_attn
        
        # Get Q, K, V projections manually
        # in_proj_weight is [3*embed_dim, embed_dim] and in_proj_bias is [3*embed_dim]
        qkv = F.linear(x, attn_module.in_proj_weight, attn_module.in_proj_bias)
        qkv = qkv.reshape(B, T, 3, self.d_model)
        q, k, v = qkv.unbind(dim=2)  # Each is [B, T, D]
        
        # Reshape for multi-head attention
        # [B, T, D] -> [B, T, H, D//H] -> [B, H, T, D//H]
        head_dim = self.d_model // self.nhead
        q = q.reshape(B, T, self.nhead, head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.nhead, head_dim).transpose(1, 2)
        
        # Compute attention scores [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # mask is [B, T] with True for padding
            # Expand to [B, 1, 1, T] for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        return attn_weights.detach()


@register_method('attention_selection')
class AttentionSelectionMethod(BaseMethod):
    """
    Attention-based token selection method for learning confidence scores from logprobs.
    
    This method:
    1. Extracts attention patterns from a 2-layer transformer
    2. Computes attention statistics per token
    3. Uses a learned importance network to score tokens
    4. Selects top-k tokens using entmax (sparse softmax)
    5. Aggregates selected tokens for final prediction
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Model architecture parameters
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.use_projection = config['use_projection']
        self.use_both_layers = config.get('use_both_layers', True)
        
        # Token selection parameters
        self.k = config.get('k', 10)
        self.temperature = config.get('temperature', 1.0)
        
        # Training parameters
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.batch_problems = config['batch_problems']
        self.margin = config['margin']
        self.patience = config['patience']
        self.early_stop_metric = config['early_stop_metric']
        
        # Loss weights
        self.sparsity_lambda = config.get('sparsity_lambda', 1e-3)
        
        # Optional parameters
        self.tail_tokens = config.get('tail_tokens', None)
        self.seed = config.get('seed', None)
        self.loss_type = config.get('loss_type', 'hinge')
        self.use_weighted_loss = config.get('use_weighted_loss', False)
        self.weight_decay = config.get('weight_decay', 0.0)
        
        self.num_top_logprobs = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _preload_data_to_gpu(self, data: List[Dict]) -> List[Dict]:
        """Pre-load all logprobs to GPU as tensors to avoid repeated CPU-GPU transfers."""
        print(f"\nPre-loading {len(data)} items to GPU...")
        sys.stdout.flush()
        t_start = time.perf_counter()
        
        preloaded_data = []
        for item in data:
            lp = item['logprobs']
            if self.tail_tokens is not None and self.tail_tokens > 0:
                lp = lp[-self.tail_tokens:]
            
            # Convert to GPU tensor immediately
            tensor = torch.tensor(lp, dtype=torch.float32, device=self.device)
            
            preloaded_item = item.copy()
            preloaded_item['logprobs_tensor'] = tensor
            preloaded_item['seq_len'] = tensor.shape[0]
            preloaded_data.append(preloaded_item)
        
        t_end = time.perf_counter()
        print(f"✓ Pre-loaded data to GPU in {t_end - t_start:.3f}s")
        sys.stdout.flush()
        
        return preloaded_data

    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Train attention selection model.

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
        print(f"{self.name}: Training attention selection model...")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        # Pre-load all data to GPU
        train_data = self._preload_data_to_gpu(train_data)
        if val_data:
            val_data = self._preload_data_to_gpu(val_data)

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
        self.model = AttentionSelectionModel(
            num_top_logprobs=self.num_top_logprobs,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_len=self.tail_tokens if self.tail_tokens else 2048,
            k=self.k,
            temperature=self.temperature,
            use_both_layers=self.use_both_layers,
            use_projection=self.use_projection
        )
        self.model.to(self.device)

        print(f"Model architecture:")
        print(f"  d_model: {self.d_model}, nhead: {self.nhead}, layers: {self.num_layers}")
        print(f"  k: {self.k}, temperature: {self.temperature}")
        print(f"  use_both_layers: {self.use_both_layers}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        sys.stdout.flush()

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_metric = -1e9
        best_epoch = -1
        epochs_no_improve = 0
        best_state = None
        history = {
            'train_loss': [], 'train_task_loss': [], 'train_avg_selected': [],
            'val_acc': [], 'val_loss': []
        }

        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_metrics = defaultdict(float)

            problem_ids = sorted(list(train_problems.keys()))
            import random
            random.shuffle(problem_ids)

            batch_loss = 0.0
            batch_count = 0
            optimizer.zero_grad()

            for i, pid in enumerate(problem_ids):
                rollouts = train_problems[pid]

                # Build batch
                tensors = []
                truths = []
                for r in rollouts:
                    tensors.append(r['logprobs_tensor'])
                    truths.append(1 if r['is_correct'] else 0)

                if not tensors:
                    continue

                # Determine max_len
                if self.tail_tokens is not None and self.tail_tokens > 0:
                    max_len = self.tail_tokens
                else:
                    max_len = max(t.shape[0] for t in tensors)
                
                # Create batch and mask directly on GPU
                batch = torch.zeros((len(tensors), max_len, tensors[0].shape[1]),
                                  dtype=torch.float32, device=self.device)
                mask = torch.ones((len(tensors), max_len), dtype=torch.bool, device=self.device)
                
                for bi, t in enumerate(tensors):
                    seq_len = min(t.shape[0], max_len)
                    batch[bi, :seq_len, :] = t[:seq_len]
                    mask[bi, :seq_len] = False
                
                # Forward pass with details
                outputs = self.model(batch, mask=mask, return_details=True)
                scores = outputs['metric']
                weights = outputs['token_weights']
                num_selected = outputs['num_selected']

                truths = torch.tensor(truths, dtype=torch.bool, device=self.device)
                pos_idx = torch.nonzero(truths).squeeze(-1)
                neg_idx = torch.nonzero(~truths).squeeze(-1)

                if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                    continue

                # Compute task loss
                if self.loss_type == 'bce':
                    targets = truths.float()
                    probs = torch.sigmoid(scores)
                    
                    if self.use_weighted_loss:
                        n_pos = pos_idx.numel()
                        n_neg = neg_idx.numel()
                        total = n_pos + n_neg
                        pos_weight = total / (2.0 * n_pos)
                        neg_weight = total / (2.0 * n_neg)
                        weights_loss = torch.where(truths, pos_weight, neg_weight)
                        task_loss = F.binary_cross_entropy(probs, targets, weight=weights_loss, reduction='sum')
                    else:
                        task_loss = F.binary_cross_entropy(probs, targets, reduction='sum')
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
                        task_loss = (losses * weight_matrix).sum()
                    else:
                        task_loss = losses.sum()

                # Simplified loss: just task loss (CLS attention is fixed, no regularization needed)
                total_loss = task_loss

                total_loss.backward()
                batch_loss += total_loss.item()
                epoch_metrics['task_loss'] += task_loss.item()
                epoch_metrics['avg_selected'] += num_selected.float().mean().item()
                batch_count += 1

                if batch_count >= self.batch_problems:
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += batch_loss
                    batch_loss = 0.0
                    batch_count = 0

            # Final optimizer step
            if batch_count > 0:
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += batch_loss

            # Evaluate
            val_acc = self._evaluate_accuracy(val_problems if val_problems else train_problems)
            val_loss = self._evaluate_loss(val_problems if val_problems else train_problems)

            history['train_loss'].append(epoch_loss)
            history['train_task_loss'].append(epoch_metrics['task_loss'])
            history['train_avg_selected'].append(epoch_metrics['avg_selected'] / len(problem_ids) if problem_ids else 0)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            avg_selected = epoch_metrics['avg_selected'] / len(problem_ids) if problem_ids else 0
            print(f"Epoch {epoch}/{self.num_epochs} - "
                  f"train_loss={epoch_loss:.4f} "
                  f"avg_selected={avg_selected:.1f} "
                  f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
            sys.stdout.flush()
            
            # Log to wandb if enabled
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': epoch_loss,
                        'train_task_loss': epoch_metrics['task_loss'],
                        'train_avg_selected': avg_selected,
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
                    tensors.append(r['logprobs_tensor'])
                    truths.append(r['is_correct'])

                if not tensors:
                    continue

                if self.tail_tokens is not None and self.tail_tokens > 0:
                    max_len = self.tail_tokens
                else:
                    max_len = max(t.shape[0] for t in tensors)
                
                batch = torch.zeros((len(tensors), max_len, tensors[0].shape[1]),
                                  dtype=torch.float32, device=self.device)
                mask = torch.ones((len(tensors), max_len), dtype=torch.bool, device=self.device)
                
                for i, t in enumerate(tensors):
                    seq_len = min(t.shape[0], max_len)
                    batch[i, :seq_len, :] = t[:seq_len]
                    mask[i, :seq_len] = False

                scores = self.model(batch, mask=mask, return_details=False)
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
                    tensors.append(r['logprobs_tensor'])
                    truths.append(1 if r['is_correct'] else 0)

                if not tensors:
                    continue

                if self.tail_tokens is not None and self.tail_tokens > 0:
                    max_len = self.tail_tokens
                else:
                    max_len = max(t.shape[0] for t in tensors)
                
                batch = torch.zeros((len(tensors), max_len, tensors[0].shape[1]),
                                  dtype=torch.float32, device=self.device)
                mask = torch.ones((len(tensors), max_len), dtype=torch.bool, device=self.device)
                
                for i, t in enumerate(tensors):
                    seq_len = min(t.shape[0], max_len)
                    batch[i, :seq_len, :] = t[:seq_len]
                    mask[i, :seq_len] = False

                scores = self.model(batch, mask=mask, return_details=False)
                truths_t = torch.tensor(truths, dtype=torch.bool, device=self.device)
                pos_idx = torch.nonzero(truths_t).squeeze(-1)
                neg_idx = torch.nonzero(~truths_t).squeeze(-1)

                if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                    continue

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
        Predict confidence using trained attention selection model.

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
        mask[0, :seq_len] = False
        
        batch = batch.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            score = self.model(batch, mask=mask, return_details=False).cpu().item()

        return score

    def save(self, output_dir: str) -> Optional[str]:
        """Save trained model."""
        if self.model is None:
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / 'attention_selection_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_top_logprobs': self.num_top_logprobs,
            'config': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'use_projection': self.use_projection,
                'use_both_layers': self.use_both_layers,
                'k': self.k,
                'temperature': self.temperature,
                'num_epochs': self.num_epochs,
                'lr': self.lr,
                'batch_problems': self.batch_problems,
                'margin': self.margin,
                'patience': self.patience,
                'early_stop_metric': self.early_stop_metric,
                'sparsity_lambda': self.sparsity_lambda,
                'tail_tokens': self.tail_tokens,
                'seed': self.seed,
                'loss_type': self.loss_type,
                'use_weighted_loss': self.use_weighted_loss,
                'weight_decay': self.weight_decay
            }
        }, model_path)
        
        return str(model_path)
    
    def load(self, model_path: str) -> None:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        config = checkpoint['config']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.use_projection = config['use_projection']
        self.use_both_layers = config.get('use_both_layers', True)
        self.k = config.get('k', 10)
        self.temperature = config.get('temperature', 1.0)
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.batch_problems = config['batch_problems']
        self.margin = config['margin']
        self.patience = config['patience']
        self.early_stop_metric = config['early_stop_metric']
        self.sparsity_lambda = config.get('sparsity_lambda', 1e-3)
        self.tail_tokens = config['tail_tokens']
        self.seed = config.get('seed')
        self.loss_type = config.get('loss_type', 'hinge')
        self.use_weighted_loss = config.get('use_weighted_loss', False)
        self.weight_decay = config.get('weight_decay', 0.0)
        
        # Load model
        self.num_top_logprobs = checkpoint['num_top_logprobs']
        self.model = AttentionSelectionModel(
            num_top_logprobs=self.num_top_logprobs,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_len=self.tail_tokens if self.tail_tokens else 2048,
            k=self.k,
            temperature=self.temperature,
            use_both_layers=self.use_both_layers,
            use_projection=self.use_projection
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.is_trained = True
