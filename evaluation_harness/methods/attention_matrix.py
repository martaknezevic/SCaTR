"""
Attention-based methods for learning from transformer attention patterns.

This module implements methods that extract and learn from attention weights,
particularly focusing on CLS token attention in the last layer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import sys
import json

from methods.base_method import BaseMethod
from methods.method_registry import register_method


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with CLS token for sequence encoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len + 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass with optional attention extraction.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Padding mask (batch, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits (batch, 1)
            attention: (Optional) Attention weights from last layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len+1, :]
        
        # Create mask for CLS token (if original mask provided)
        if mask is not None:
            # Add False for CLS token (not masked)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Transformer encoding
        if return_attention:
            # We need to manually extract attention from the last layer
            # Store intermediate outputs
            cls_attention = None
            for i, layer in enumerate(self.transformer.layers):
                if i == len(self.transformer.layers) - 1:
                    # Last layer - extract attention
                    # We need to manually call self_attn with need_weights=True
                    # First apply layer norm (since norm_first=True)
                    x_norm = layer.norm1(x)
                    
                    # Call self-attention directly with need_weights=True
                    attn_output, attn_weights = layer.self_attn(
                        x_norm, x_norm, x_norm,
                        key_padding_mask=mask,
                        need_weights=True,
                        average_attn_weights=False  # Get per-head attention
                    )
                    
                    # Complete the layer forward pass manually
                    x = x + layer.dropout1(attn_output)
                    x = x + layer._ff_block(layer.norm2(x))
                    
                    # Extract CLS attention from the last layer
                    if attn_weights is not None:
                        # attn_weights: (batch, num_heads, seq_len+1, seq_len+1)
                        # Extract attention from CLS token (first token) to all others
                        cls_attention = attn_weights[:, :, 0, :]  # (batch, num_heads, seq_len+1)
                else:
                    x = layer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x, src_key_padding_mask=mask)
            cls_attention = None
        
        # Extract CLS token representation
        cls_output = x[:, 0, :]  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(cls_output)
        
        if return_attention:
            return logits, cls_attention
        else:
            return logits


class CLSAttentionExtractor:
    """
    Extracts raw attention weights from the last layer's CLS token.
    
    This class handles:
    - Single-head and multi-head attention matrices
    - Returns attention distribution over all tokens in the sequence
    """
    
    def __init__(self, model: TransformerEncoder):
        self.model = model
    
    def extract_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract CLS token attention from the last transformer layer.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Padding mask (batch, seq_len)
            
        Returns:
            Attention weights (batch, seq_len) - attention from CLS to all tokens
        """
        self.model.eval()
        
        with torch.no_grad():
            _, cls_attention = self.model(x, mask=mask, return_attention=True)
        
        if cls_attention is None:
            raise ValueError("Failed to extract attention weights")
        
        # cls_attention shape: (batch, num_heads, seq_len+1)
        # Average over heads and remove CLS token itself
        if cls_attention.dim() == 3:
            # Multi-head: average across heads
            avg_attention = cls_attention.mean(dim=1)  # (batch, seq_len+1)
        else:
            # Single head
            avg_attention = cls_attention
        
        # Remove attention to CLS token itself (first position)
        attention_to_tokens = avg_attention[:, 1:]  # (batch, seq_len)
        
        return attention_to_tokens


class TopKAttentionMetricLearner(nn.Module):
    """
    Neural network that learns from top-k attended tokens.
    
    Takes as input:
    - Top-k attention weights
    - Corresponding logprobs for those tokens
    
    Computes statistics and uses MLP to predict confidence scores.
    """
    
    def __init__(
        self,
        k: int,
        num_logprobs: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.k = k
        self.num_logprobs = num_logprobs
        
        # Feature size: k attention weights + k * num_logprobs + statistics
        # Statistics: mean, max, min of attention and logprobs
        feature_size = k + k * num_logprobs + 6  # 6 statistics
        
        # 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, attention_weights, logprobs):
        """
        Forward pass.
        
        Args:
            attention_weights: Top-k attention weights (batch, k)
            logprobs: Logprobs for top-k tokens (batch, k, num_logprobs)
            
        Returns:
            Confidence scores (batch, 1)
        """
        batch_size = attention_weights.shape[0]
        
        # Flatten attention weights and logprobs
        attention_flat = attention_weights.reshape(batch_size, -1)  # (batch, k)
        logprobs_flat = logprobs.reshape(batch_size, -1)  # (batch, k * num_logprobs)
        
        # Compute statistics
        # attention_weights is (batch, k), so we compute stats along dim=1
        attn_mean = attention_weights.mean(dim=1, keepdim=True)  # (batch, 1)
        attn_max = attention_weights.max(dim=1, keepdim=True)[0]  # (batch, 1)
        attn_min = attention_weights.min(dim=1, keepdim=True)[0]  # (batch, 1)
        
        # logprobs is (batch, k, num_logprobs)
        logprobs_mean = logprobs.mean(dim=(1, 2), keepdim=True).reshape(batch_size, 1)  # (batch, 1)
        logprobs_max = logprobs.max(dim=2)[0].max(dim=1, keepdim=True)[0]  # (batch, 1)
        logprobs_min = logprobs.min(dim=2)[0].min(dim=1, keepdim=True)[0]  # (batch, 1)
        
        # Concatenate all features (all should be 2D: batch, features)
        features = torch.cat([
            attention_flat,
            logprobs_flat,
            attn_mean,
            attn_max,
            attn_min,
            logprobs_mean,
            logprobs_max,
            logprobs_min
        ], dim=1)
        
        # Predict confidence
        confidence = self.mlp(features)
        
        return confidence


@register_method('attention_matrix')
class CLSAttentionTopKMetricMethod(BaseMethod):
    """
    Method that learns from CLS attention and top-k attended tokens.
    
    Two-stage training:
    1. Train transformer encoder with binary classification
    2. Freeze transformer, extract CLS attention, train metric learner on top-k tokens
    
    Key features:
    - Extracts attention from last layer's CLS token
    - Selects top-k tokens with highest attention weights
    - Learns NN based on attention weights AND logprobs of those tokens
    - Early stopping for both stages
    - Full save/load functionality
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        print(config)
        # Architecture parameters
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.top_k = config.get('top_k', 10)
        self.metric_hidden_dim = config.get('metric_hidden_dim', 64)
        
        # Training parameters - Stage 1 (transformer)
        self.stage1_lr = config.get('stage1_lr', 0.001)
        self.stage1_weight_decay = config.get('stage1_weight_decay', 0.0001)
        self.stage1_epochs = config.get('stage1_epochs', 10)
        self.stage1_patience = config.get('stage1_patience', 10)
        self.stage1_early_stopping_metric = config.get('stage1_early_stopping_metric', 'val_acc')  # 'val_acc' or 'val_loss'
        
        # Training parameters - Stage 2 (metric learner)
        self.stage2_lr = config.get('stage2_lr', 0.0005)
        self.stage2_weight_decay = config.get('stage2_weight_decay', 0.0001)
        self.stage2_epochs = config.get('stage2_epochs', 10)
        self.stage2_patience = config.get('stage2_patience', 10)
        self.stage2_early_stopping_metric = config.get('stage2_early_stopping_metric', 'val_acc')  # 'val_acc' or 'val_loss'
        
        # Common training parameters
        self.batch_size = config.get('batch_size', 32)
        self.loss_type = config.get('loss_type', 'bce')  # 'bce' or 'hinge'
        self.margin = config.get('margin', 1.0)  # Margin for hinge loss
        
        # Data parameters
        self.tail_tokens = config.get('tail_tokens', None)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.transformer = None
        self.metric_learner = None
        self.attention_extractor = None
        self.input_dim = None
    
    def _extract_features(self, rollout: Dict) -> np.ndarray:
        """Extract logprobs features from rollout."""
        logprobs = rollout['logprobs']
        
        if logprobs.size == 0:
            return np.array([])
        
        # Select tail tokens if specified
        if self.tail_tokens is not None and self.tail_tokens > 0:
            logprobs = logprobs[-self.tail_tokens:]
        
        return logprobs
    
    def _preload_data_to_gpu(self, data: List[Dict]) -> List[Dict]:
        """Pre-load all logprobs to GPU as tensors to avoid repeated CPU-GPU transfers."""
        print(f"Pre-loading {len(data)} items to GPU...")
        sys.stdout.flush()
        
        preloaded_data = []
        for item in data:
            features = self._extract_features(item)
            
            if features.size == 0:
                continue
            
            # Convert to GPU tensor immediately
            tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            
            preloaded_item = item.copy()
            preloaded_item['logprobs_tensor'] = tensor
            preloaded_item['seq_len'] = tensor.shape[0]
            preloaded_data.append(preloaded_item)
        
        print(f"✓ Pre-loaded data to GPU")
        sys.stdout.flush()
        
        return preloaded_data
    
    def _create_transformer(self, input_dim: int) -> TransformerEncoder:
        """Create transformer encoder."""
        model = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        return model.to(self.device)
    
    def _create_metric_learner(self, num_logprobs: int) -> TopKAttentionMetricLearner:
        """Create metric learner."""
        model = TopKAttentionMetricLearner(
            k=self.top_k,
            num_logprobs=num_logprobs,
            hidden_dim=self.metric_hidden_dim,
            dropout=self.dropout
        )
        return model.to(self.device)
    
    def _train_stage1(
        self,
        train_features: List[np.ndarray],
        train_labels: List[int],
        val_features: Optional[List[np.ndarray]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Train transformer encoder with binary classification.
        
        Returns:
            Training metrics
        """
        print(f"\n{'='*80}")
        print(f"STAGE 1: Training Transformer Encoder")
        print(f"{'='*80}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Num heads: {self.num_heads}")
        print(f"Num layers: {self.num_layers}")
        print(f"Learning rate: {self.stage1_lr}")
        print(f"Epochs: {self.stage1_epochs}")
        sys.stdout.flush()
        
        # Create transformer
        self.transformer = self._create_transformer(self.input_dim)
        
        # Optimizer
        optimizer = optim.AdamW(
            self.transformer.parameters(),
            lr=self.stage1_lr,
            weight_decay=self.stage1_weight_decay
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.stage1_epochs):
            self.transformer.train()
            epoch_losses = []
            
            # Shuffle
            indices = np.random.permutation(len(train_features))
            
            # Mini-batch training
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Prepare batch
                batch_features = [train_features[idx] for idx in batch_indices]
                batch_labels = [train_labels[idx] for idx in batch_indices]
                
                # Pad sequences (features are already GPU tensors)
                max_len = max(f.shape[0] for f in batch_features)
                padded_features = []
                masks = []
                
                for features in batch_features:
                    seq_len = features.shape[0]
                    if seq_len < max_len:
                        padding = torch.zeros(max_len - seq_len, self.input_dim, device=self.device)
                        padded = torch.cat([features, padding], dim=0)
                        mask = torch.tensor([False] * seq_len + [True] * (max_len - seq_len), dtype=torch.bool, device=self.device)
                    else:
                        padded = features
                        mask = torch.tensor([False] * seq_len, dtype=torch.bool, device=self.device)
                    
                    padded_features.append(padded)
                    masks.append(mask)
                
                # Stack tensors (already on GPU)
                X = torch.stack(padded_features)
                y = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)
                mask = torch.stack(masks)
                
                # Forward
                logits = self.transformer(X, mask=mask)
                
                # Compute loss based on loss_type
                if self.loss_type == 'bce':
                    # Binary cross-entropy loss
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y)
                else:  # hinge
                    # Hinge loss: separate positive and negative samples
                    scores = logits.squeeze()
                    pos_mask = y == 1
                    neg_mask = y == 0
                    
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        pos_scores = scores[pos_mask].unsqueeze(1)  # (n_pos, 1)
                        neg_scores = scores[neg_mask].unsqueeze(0)  # (1, n_neg)
                        # Pairwise hinge loss
                        losses = F.relu(self.margin - (pos_scores - neg_scores))
                        loss = losses.mean()
                    else:
                        # Skip batch if all positive or all negative
                        continue
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            # Validation
            val_acc = None
            if val_features is not None and val_labels is not None:
                val_acc = self._evaluate_stage1(val_features, val_labels)
                val_accuracies.append(val_acc)
                
                print(f"Epoch {epoch+1}/{self.stage1_epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2%}")
                
                # Log to wandb if enabled
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            'stage1_epoch': epoch + 1,
                            'stage1_train_loss': avg_loss,
                            'stage1_val_acc': val_acc
                        })
                except ImportError:
                    pass
                
                # Early stopping based on selected metric
                if self.stage1_early_stopping_metric == 'val_loss':
                    current_metric = avg_loss
                    is_better = current_metric < best_val_acc
                else:  # val_acc
                    current_metric = val_acc
                    is_better = current_metric > best_val_acc
                
                if is_better:
                    best_val_acc = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.stage1_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.stage1_epochs} - Loss: {avg_loss:.4f}")
            
            sys.stdout.flush()
        
        print(f"\n✓ Stage 1 complete")
        sys.stdout.flush()
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def _evaluate_stage1(self, features: List[np.ndarray], labels: List[int]) -> float:
        """Evaluate transformer encoder accuracy."""
        self.transformer.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(features), self.batch_size):
                batch_features = features[i:i + self.batch_size]
                batch_labels = labels[i:i + self.batch_size]
                
                # Pad sequences (features are already GPU tensors)
                max_len = max(f.shape[0] for f in batch_features)
                padded_features = []
                masks = []
                
                for f in batch_features:
                    seq_len = f.shape[0]
                    if seq_len < max_len:
                        padding = torch.zeros(max_len - seq_len, self.input_dim, device=self.device)
                        padded = torch.cat([f, padding], dim=0)
                        mask = torch.tensor([False] * seq_len + [True] * (max_len - seq_len), dtype=torch.bool, device=self.device)
                    else:
                        padded = f
                        mask = torch.tensor([False] * seq_len, dtype=torch.bool, device=self.device)
                    
                    padded_features.append(padded)
                    masks.append(mask)
                
                X = torch.stack(padded_features)
                y = torch.tensor(batch_labels, dtype=torch.float32, device=self.device)
                mask = torch.stack(masks)
                
                logits = self.transformer(X, mask=mask)
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                
                correct += (predictions == y).sum().item()
                total += len(batch_labels)
        
        return correct / total if total > 0 else 0.0
    
    def _extract_topk_features(
        self,
        features: torch.Tensor,
        attention: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract top-k features based on attention weights.
        
        Args:
            features: Logprobs (seq_len, num_logprobs) - GPU tensor
            attention: Attention weights (seq_len,) - GPU tensor
            
        Returns:
            topk_attention: Top-k attention weights (k,) - GPU tensor
            topk_logprobs: Logprobs for top-k tokens (k, num_logprobs) - GPU tensor
        """
        seq_len = attention.shape[0]
    
        # Get top-k indices based on attention weights
        k = min(seq_len, self.top_k)
        _, topk_indices = torch.topk(attention, k)
        
        # Extract top-k features
        topk_attention = attention[topk_indices]
        topk_logprobs = features[topk_indices]
        
        # Pad if necessary
        if k < self.top_k:
            # Pad with zeros
            attn_padding = torch.zeros(self.top_k - k, device=self.device)
            logprob_padding = torch.zeros(self.top_k - k, features.shape[1], device=self.device)
            
            topk_attention = torch.cat([topk_attention, attn_padding])
            topk_logprobs = torch.cat([topk_logprobs, logprob_padding], dim=0)
        
        return topk_attention, topk_logprobs
    
    def _train_stage2(
        self,
        train_data: List[Dict],
        train_labels: List[int],
        val_data: Optional[List[Dict]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Stage 2: Freeze transformer, extract attention, train metric learner.
        
        Returns:
            Training metrics
        """
        print(f"\n{'='*80}")
        print(f"STAGE 2: Training Metric Learner on Top-K Attention")
        print(f"{'='*80}")
        print(f"Top K: {self.top_k}")
        print(f"Metric hidden dim: {self.metric_hidden_dim}")
        print(f"Learning rate: {self.stage2_lr}")
        print(f"Epochs: {self.stage2_epochs}")
        sys.stdout.flush()
        
        # Freeze transformer
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.transformer.eval()
        
        # Create attention extractor
        self.attention_extractor = CLSAttentionExtractor(self.transformer)
        
        # Extract attention for all training data
        print("\nExtracting attention weights...")
        sys.stdout.flush()
        
        train_topk_attention = []
        train_topk_logprobs = []
        
        for item in train_data:
            # Use pre-loaded tensor if available, otherwise create on-the-fly
            if 'logprobs_tensor' in item:
                features_tensor = item['logprobs_tensor']
                X = features_tensor.unsqueeze(0)
            else:
                features = self._extract_features(item)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
                X = features_tensor.unsqueeze(0)
            
            # Get attention weights (keep on GPU)
            attention = self.attention_extractor.extract_attention(X).squeeze(0)
            
            # Extract top-k (all on GPU)
            topk_attn, topk_lp = self._extract_topk_features(features_tensor, attention)
            train_topk_attention.append(topk_attn)
            train_topk_logprobs.append(topk_lp)
        
        # Create metric learner
        num_logprobs = train_topk_logprobs[0].shape[1]
        self.metric_learner = self._create_metric_learner(num_logprobs)
        
        # Optimizer
        optimizer = optim.AdamW(
            self.metric_learner.parameters(),
            lr=self.stage2_lr,
            weight_decay=self.stage2_weight_decay
        )
        
        # Training loop
        best_val_acc = float('inf') if self.stage2_early_stopping_metric == 'val_loss' else 0.0
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.stage2_epochs):
            self.metric_learner.train()
            epoch_losses = []
            
            # Shuffle
            indices = np.random.permutation(len(train_topk_attention))
            
            # Mini-batch training
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Stack tensors (already on GPU)
                attn = torch.stack([train_topk_attention[idx] for idx in batch_indices])
                lp = torch.stack([train_topk_logprobs[idx] for idx in batch_indices])
                y = torch.tensor([train_labels[idx] for idx in batch_indices], dtype=torch.float32, device=self.device)
                
                # Forward
                logits = self.metric_learner(attn, lp)
                
                # Compute loss based on loss_type
                if self.loss_type == 'bce':
                    # Binary cross-entropy loss
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y)
                else:  # hinge
                    # Hinge loss: separate positive and negative samples
                    scores = logits.squeeze()
                    pos_mask = y == 1
                    neg_mask = y == 0
                    
                    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                        pos_scores = scores[pos_mask].unsqueeze(1)  # (n_pos, 1)
                        neg_scores = scores[neg_mask].unsqueeze(0)  # (1, n_neg)
                        # Pairwise hinge loss
                        losses = F.relu(self.margin - (pos_scores - neg_scores))
                        loss = losses.mean()
                    else:
                        # Skip batch if all positive or all negative
                        continue
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            # Validation
            val_acc = None
            if val_data is not None and val_labels is not None:
                val_acc = self._evaluate_stage2(val_data, val_labels)
                val_accuracies.append(val_acc)
                
                print(f"Epoch {epoch+1}/{self.stage2_epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2%}")
                
                # Log to wandb if enabled
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            'stage2_epoch': epoch + 1,
                            'stage2_train_loss': avg_loss,
                            'stage2_val_acc': val_acc
                        })
                except ImportError:
                    pass
                
                # Early stopping based on selected metric
                if self.stage2_early_stopping_metric == 'val_loss':
                    current_metric = avg_loss
                    is_better = current_metric < best_val_acc
                else:  # val_acc
                    current_metric = val_acc
                    is_better = current_metric > best_val_acc
                
                if is_better:
                    best_val_acc = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.stage2_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.stage2_epochs} - Loss: {avg_loss:.4f}")
            
            sys.stdout.flush()
        
        print(f"\n✓ Stage 2 complete")
        sys.stdout.flush()
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def _evaluate_stage2(self, val_data: List[Dict], labels: List[int]) -> float:
        """Evaluate metric learner accuracy."""
        self.metric_learner.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(len(val_data)):
                # Use pre-loaded tensor if available, otherwise create on-the-fly
                if 'logprobs_tensor' in val_data[i]:
                    features_tensor = val_data[i]['logprobs_tensor']
                    X = features_tensor.unsqueeze(0)
                else:
                    f = self._extract_features(val_data[i])
                    features_tensor = torch.tensor(f, dtype=torch.float32, device=self.device)
                    X = features_tensor.unsqueeze(0)
                
                label = labels[i]
                
                # Get attention (keep on GPU)
                attention = self.attention_extractor.extract_attention(X).squeeze(0)
                
                # Extract top-k (all on GPU)
                topk_attn, topk_lp = self._extract_topk_features(features_tensor, attention)
                
                # Add batch dimension
                attn = topk_attn.unsqueeze(0)
                lp = topk_lp.unsqueeze(0)
                y = label
                
                # Forward
                logits = self.metric_learner(attn, lp)
                prediction = (torch.sigmoid(logits.squeeze()) > 0.5).float().item()
                
                if prediction == y:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def train(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Two-stage training process.
        
        Stage 1: Train transformer encoder
        Stage 2: Freeze transformer, train metric learner on top-k attention
        """
        print(f"\n{'='*80}")
        print(f"Training {self.name}")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        # Pre-load data to GPU
        print("\nPre-loading data to GPU...")
        train_data = self._preload_data_to_gpu(train_data)
        if val_data is not None:
            val_data = self._preload_data_to_gpu(val_data)
        
        # Extract features for Stage 1
        print("\nPreparing features for Stage 1...")
        train_features = []
        train_labels = []
        
        for item in train_data:
            if 'logprobs_tensor' in item:
                train_features.append(item['logprobs_tensor'])
                train_labels.append(1 if item['is_correct'] else 0)
        
        val_features = None
        val_labels = None
        if val_data is not None:
            val_features = []
            val_labels = []
            for item in val_data:
                if 'logprobs_tensor' in item:
                    val_features.append(item['logprobs_tensor'])
                    val_labels.append(1 if item['is_correct'] else 0)
        
        # Determine input dimension
        self.input_dim = train_features[0].shape[1]
        print(f"Input dimension: {self.input_dim}")
        print(f"Number of training samples: {len(train_features)}")
        sys.stdout.flush()
        
        # Stage 1: Train transformer
        stage1_metrics = self._train_stage1(
            train_features, train_labels,
            val_features, val_labels
        )
        
        # Stage 2: Train metric learner (pass full data items for pre-loaded tensors)
        stage2_metrics = self._train_stage2(
            train_data, train_labels,
            val_data, val_labels
        )
        
        self.is_trained = True
        
        print(f"\n✓ Training complete")
        sys.stdout.flush()
        
        return {
            'stage1': stage1_metrics,
            'stage2': stage2_metrics
        }
    
    def predict_confidence(self, rollout: Dict) -> float:
        """Predict confidence using metric learner on top-k attention."""
        if self.metric_learner is None or self.attention_extractor is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Use pre-loaded tensor if available, otherwise create on-the-fly
        if 'logprobs_tensor' in rollout:
            features_tensor = rollout['logprobs_tensor']
            X = features_tensor.unsqueeze(0)
        else:
            features = self._extract_features(rollout)
            
            if features.size == 0:
                return 0.0
            
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            X = features_tensor.unsqueeze(0)
        
        # Get attention (keep on GPU)
        attention = self.attention_extractor.extract_attention(X).squeeze(0)
        
        # Extract top-k (all on GPU)
        topk_attn, topk_lp = self._extract_topk_features(features_tensor, attention)
        
        # Add batch dimension
        attn = topk_attn.unsqueeze(0)
        lp = topk_lp.unsqueeze(0)
        
        # Predict
        self.metric_learner.eval()
        with torch.no_grad():
            confidence = torch.sigmoid(self.metric_learner(attn, lp)).item()
        
        return confidence
    
    def save_model(self, output_dir: str) -> Optional[str]:
        """Save both transformer and metric learner."""
        if self.transformer is None or self.metric_learner is None:
            return None
        
        model_path = Path(output_dir) / 'cls_attention_topk_model.pt'
        torch.save({
            'transformer_state_dict': self.transformer.state_dict(),
            'metric_learner_state_dict': self.metric_learner.state_dict(),
            'input_dim': self.input_dim,
            'config': self.config
        }, model_path)
        
        return str(model_path)
    
    def load_model(self, model_path: str):
        """Load both transformer and metric learner."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.input_dim = checkpoint['input_dim']
        
        # Recreate models
        self.transformer = self._create_transformer(self.input_dim)
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.transformer.eval()
        
        # Need to determine num_logprobs from saved state
        # We can infer it from the metric learner's first layer size
        metric_state = checkpoint['metric_learner_state_dict']
        first_layer_size = metric_state['mlp.0.weight'].shape[1]
        # first_layer_size = k + k * num_logprobs + 6
        # Solve for num_logprobs
        num_logprobs = (first_layer_size - self.top_k - 6) // self.top_k
        
        self.metric_learner = self._create_metric_learner(num_logprobs)
        self.metric_learner.load_state_dict(metric_state)
        self.metric_learner.eval()
        
        self.attention_extractor = CLSAttentionExtractor(self.transformer)
        
        self.is_trained = True
