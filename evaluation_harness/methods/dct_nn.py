"""
DCT-based Neural Network method for learning from logprobs.

Uses Discrete Cosine Transform (DCT) to extract frequency-domain features
from logprob sequences, then trains a neural network on these features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from scipy.fftpack import dct
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys

from methods.base_method import BaseMethod
from methods.method_registry import register_method


class DCTNeuralNet(nn.Module):
    """Neural network that operates on DCT coefficients."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


@register_method('dct_nn')
class DCTNNMethod(BaseMethod):
    """
    DCT-based neural network method for learning confidence scores.
    
    Extracts DCT coefficients from logprob sequences and trains a neural network
    to predict correct vs incorrect sequences.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Required parameters
        self.K = config.get('K', 128)  # Number of DCT coefficients
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.batch_problems = config['batch_problems']
        self.margin = config['margin']
        self.patience = config['patience']
        self.early_stop_metric = config.get('early_stop_metric', 'val_acc')
        
        # Optional parameters
        self.tail_tokens = config.get('tail_tokens', None)
        self.seed = config.get('seed', None)
        self.loss_type = config.get('loss_type', 'hinge')  # 'hinge' or 'bce'
        self.use_weighted_loss = config.get('use_weighted_loss', False)
        self.normalize_dct = config.get('normalize_dct', False)  # Remove DC bias before DCT
        self.choice_logprobs = config.get('choice_logprobs', 'top1')  # 'top1' or 'mean'        
        self.window_size = config.get('window_size', 128)  # Window size for moving average        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _preload_data_to_gpu(self, data: List[Dict]) -> List[Dict]:
        """Pre-load all DCT features to GPU as tensors to avoid repeated CPU-GPU transfers."""
        print(f"Pre-loading {len(data)} items to GPU...")
        sys.stdout.flush()
        
        preloaded_data = []
        for item in data:
            # dct_features are already numpy arrays, convert to GPU tensors
            tensor = torch.tensor(item['dct_features'], dtype=torch.float32, device=self.device)
            
            preloaded_item = item.copy()
            preloaded_item['dct_features_tensor'] = tensor
            preloaded_data.append(preloaded_item)
        
        print(f"✓ Pre-loaded data to GPU")
        sys.stdout.flush()
        
        return preloaded_data
    
    def extract_dct_coeffs(self, logprobs: np.ndarray) -> np.ndarray:
        """
        Extract DCT coefficients from logprob sequence.
        
        Args:
            logprobs: Array of shape (seq_len, num_top_logprobs)
        
        Returns:
            DCT coefficients of shape (K,)
        """
        # Take first logprob from each token (top-1)
        if self.choice_logprobs == 'top1':
            seq = logprobs[:, 0]
        elif self.choice_logprobs == 'mean':
            seq = np.mean(logprobs, axis=1)
        else:
            raise ValueError(f"Unsupported choice_logprobs value: {self.choice_logprobs}")
        
        # Apply tail_tokens if specified
        if self.tail_tokens is not None and self.tail_tokens > 0:
            seq = seq[-self.tail_tokens:]
        
        # Apply windowed mean (moving average) using efficient convolution
        if self.window_size > 1 and len(seq) >= self.window_size:
            seq = np.convolve(seq, np.ones(self.window_size) / self.window_size, mode='valid')
        
        # Remove DC bias if normalize_dct is enabled
        if self.normalize_dct:
            seq = seq - seq.mean()
        
        # Apply DCT
        coeffs = dct(seq, norm='ortho')
        
        # Return first K coefficients, padding with zeros if necessary
        if len(coeffs) >= self.K:
            return coeffs[:self.K]
        else:
            # Pad with zeros if sequence is shorter than K
            padded = np.zeros(self.K)
            padded[:len(coeffs)] = coeffs
            return padded
    
    def prepare_data(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None, 
                     test_data: Optional[List[Dict]] = None):
        """
        Prepare data by extracting DCT coefficients.
        
        Args:
            train_data: Training data with logprobs
            val_data: Validation data (optional)
            test_data: Test data (optional)
        
        Returns:
            Tuple of (train_data, val_data, test_data) with DCT features
        """
        def process_data(data: List[Dict]) -> List[Dict]:
            processed = []
            for item in data:
                dct_features = self.extract_dct_coeffs(item['logprobs'])
                processed_item = item.copy()
                processed_item['dct_features'] = dct_features
                processed.append(processed_item)
            return processed
        
        print(f"\n{self.name}: Extracting DCT coefficients (K={self.K})...")
        
        train_processed = process_data(train_data) if train_data else []
        val_processed = process_data(val_data) if val_data else []
        test_processed = process_data(test_data) if test_data else []
        
        print(f"  Train: {len(train_processed)} items")
        if val_data:
            print(f"  Val: {len(val_processed)} items")
        if test_data:
            print(f"  Test: {len(test_processed)} items")
        
        return train_processed, val_processed, test_processed
    
    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Train DCT neural network model.
        
        Args:
            train_data: Training data with DCT features
            val_data: Validation data for early stopping
        
        Returns:
            Training metrics
        """
        # Set seed for reproducibility
        if self.seed is not None:
            self.set_seed(self.seed)
            print(f"Set random seed to {self.seed} for weight initialization")
        
        print(f"\n{'='*80}")
        print(f"{self.name}: Training DCT-based neural network...")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        # Pre-load data to GPU
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
        
        # Create model
        self.model = DCTNeuralNet(
            input_dim=self.K,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.model.to(self.device)
        
        print(f"Model architecture:")
        print(f"  Input dim: {self.K}, Hidden dim: {self.hidden_dim}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        sys.stdout.flush()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        best_val_metric = -1e9
        best_epoch = -1
        epochs_no_improve = 0
        best_state = None
        history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_loss': []}
        
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            problem_ids = sorted(list(train_problems.keys()))
            import random
            random.shuffle(problem_ids)
            
            batch_loss = 0.0
            batch_count = 0
            optimizer.zero_grad()
            
            for i, pid in enumerate(problem_ids):
                rollouts = train_problems[pid]
                
                # Build batch from pre-loaded GPU tensors
                features = []
                truths = []
                for r in rollouts:
                    features.append(r['dct_features_tensor'])  # Already on GPU
                    truths.append(1 if r['is_correct'] else 0)
                
                if not features:
                    continue
                
                batch = torch.stack(features)
                scores = self.model(batch)
                
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
                        n_pos = pos_idx.numel()
                        n_neg = neg_idx.numel()
                        total = n_pos + n_neg
                        pos_weight = total / (2.0 * n_pos)
                        neg_weight = total / (2.0 * n_neg)
                        weights = torch.where(truths, pos_weight, neg_weight)
                        loss = F.binary_cross_entropy(probs, targets, weight=weights, reduction='sum')
                    else:
                        loss = F.binary_cross_entropy(probs, targets, reduction='sum')
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
                        loss = (losses * weight_matrix).sum()
                    else:
                        loss = losses.sum()
                
                loss.backward()
                batch_loss += loss.item()
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
            
            # Evaluate on validation set only
            val_acc = self._evaluate_accuracy(val_problems if val_problems else train_problems)
            val_loss = self._evaluate_loss(val_problems if val_problems else train_problems)
            
            history['train_loss'].append(epoch_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch}/{self.num_epochs} - train_loss={epoch_loss:.4f} "
                  f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
            sys.stdout.flush()
            
            # Log to wandb if enabled
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': epoch_loss,
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
                features = []
                truths = []
                for r in rollouts:
                    features.append(r['dct_features_tensor'])  # Already on GPU
                    truths.append(r['is_correct'])
                
                if not features:
                    continue
                
                batch = torch.stack(features)
                scores = self.model(batch)
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
                features = []
                truths = []
                for r in rollouts:
                    features.append(r['dct_features_tensor'])  # Already on GPU
                    truths.append(1 if r['is_correct'] else 0)
                
                if not features:
                    continue
                
                batch = torch.stack(features)
                scores = self.model(batch)
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
        Predict confidence using trained model.
        
        Args:
            data_item: Dictionary with 'dct_features' key
        
        Returns:
            Confidence score
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Method must be trained before prediction")
        
        self.model.eval()
        
        # Check if already pre-loaded as tensor
        if 'dct_features_tensor' in data_item:
            features = data_item['dct_features_tensor'].unsqueeze(0)
        else:
            features = torch.tensor(data_item['dct_features'], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            score = self.model(features).cpu().item()
        
        return score
    
    def save_model(self, output_dir: str) -> Optional[str]:
        """Save trained model (framework interface)."""
        if self.model is None:
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / 'dct_nn_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'K': self.K,
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout,
                'num_epochs': self.num_epochs,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'batch_problems': self.batch_problems,
                'margin': self.margin,
                'patience': self.patience,
                'early_stop_metric': self.early_stop_metric,
                'tail_tokens': self.tail_tokens,
                'seed': self.seed,
                'loss_type': self.loss_type,
                'use_weighted_loss': self.use_weighted_loss,
                'normalize_dct': self.normalize_dct,
                'choice_logprobs': self.choice_logprobs,
                'window_size': self.window_size
            }
        }, model_path)
        
        return str(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model (framework interface)."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        config = checkpoint['config']
        self.K = config['K']
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.batch_problems = config['batch_problems']
        self.margin = config['margin']
        self.patience = config['patience']
        self.early_stop_metric = config['early_stop_metric']
        self.tail_tokens = config['tail_tokens']
        self.seed = config.get('seed')
        self.loss_type = config.get('loss_type', 'hinge')
        self.use_weighted_loss = config.get('use_weighted_loss', False)
        self.normalize_dct = config.get('normalize_dct', True)
        self.choice_logprobs = config.get('choice_logprobs', 'top1')
        self.window_size = config.get('window_size', 128)
        
        # Load model
        self.model = DCTNeuralNet(
            input_dim=self.K,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.is_trained = True
        
        print(f"\n✓ {self.name} loaded from {model_path}")
