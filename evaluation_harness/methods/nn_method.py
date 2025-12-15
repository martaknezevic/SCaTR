"""
Neural network method for learning metric weights from JSONL metrics.

Based on optimize_metric_weight_nn.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from methods.base_method import BaseMethod
from methods.method_registry import register_method



class ConfidenceNet(nn.Module):
    """Small neural network to learn confidence scores from metrics."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

@register_method('neural_net')
class NeuralNetMethod(BaseMethod):
    """
    Neural network method that learns from JSONL metrics.

    Note: This method uses metrics from JSONL file, not logprobs.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Required parameters from config
        self.hidden_dim = config['hidden_dim']
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.margin = config['margin']
        
        # Optional parameters
        self.dropout = config.get('dropout', 0.0)
        self.seed = config.get('seed', None)
        self.patience = config.get('patience', 5)  # Early stopping patience
        self.eval_interval = config.get('eval_interval', 100)  # Evaluate every N epochs
        
        # Metrics file will be set during prepare_data
        self.metrics_file = None
        self.metric_keys = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(
        self,
        train_data: Optional[List[Dict]],
        val_data: Optional[List[Dict]],
        test_data: List[Dict]
    ):
        """Prepare data and set metrics_file path from config, then load metrics for all splits."""
        import os
        
        # Get train_data_path from config to construct metrics_file path
        train_data_path = self.config.get('train_data_path')
        
        if train_data_path is None:
            raise ValueError("neural_net method requires train_data_path to locate metrics file")
        
        # Set metrics file path for training data
        self.metrics_file = Path(os.path.join(train_data_path, "all_response_metrics.jsonl"))
        
        if not self.metrics_file.exists():
            raise ValueError(f"Metrics file not found: {self.metrics_file}")
        
        # Load metrics for train and val data
        enriched_train = None
        enriched_val = None
        enriched_test = None
        
        if train_data is not None:
            enriched_train = self._load_metrics_data(train_data)
        
        if val_data is not None:
            enriched_val = self._load_metrics_data(val_data)
        
        # For test data, the metrics_file will be dynamically set by evaluation_harness
        # based on which test set is being evaluated (internal or external)
        if test_data is not None:
            enriched_test = self._load_metrics_data(test_data)
        
        return enriched_train, enriched_val, enriched_test

    def _load_metrics_data(self, data: List[Dict]) -> List[Dict]:
        """
        Load metrics from JSONL file and match with data items.

        Returns:
            List of data items with 'features' added
        """
        print(f"\n{self.name}: Loading metrics from {self.metrics_file}...")
        sys.stdout.flush()

        # Load all metrics
        metrics_dict = {}
        with open(self.metrics_file, 'r') as f:
            for line in f:
                data_rollout = json.loads(line)

                # Extract numeric metrics on first iteration
                if self.metric_keys is None:
                    excluded = ['rollout_idx', 'is_correct', 'extracted_answer', 'expected_answer', 'problem_id']
                    self.metric_keys = [k for k, v in data_rollout.items()
                                      if isinstance(v, (int, float)) and k not in excluded]
                    print(f"Found {len(self.metric_keys)} numeric metrics: {self.metric_keys}")
                    sys.stdout.flush()

                key = (data_rollout['problem_id'], data_rollout['rollout_idx'])
                metrics_dict[key] = data_rollout
        # Match metrics with data items
        enriched_data = []
        for item in data:
            key = (item['problem_id'], item['rollout_idx'])
            if key in metrics_dict:
                features = np.array([metrics_dict[key][k] for k in self.metric_keys], dtype=np.float32)
                enriched_data.append({
                    **item,
                    'features': features
                })

        print(f"Matched {len(enriched_data)} data items with metrics")
        sys.stdout.flush()

        return enriched_data

    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Train neural network using max-margin objective.

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            Training metrics
        """
        # Set seed for reproducibility
        if self.seed is not None:
            self.set_seed(self.seed)
            print(f"Set random seed to {self.seed} for weight initialization")
        
        print(f"\n{'='*80}")
        print(f"{self.name}: Training neural network...")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Load metrics for train and val data
        train_data = self._load_metrics_data(train_data)
        if val_data:
            val_data = self._load_metrics_data(val_data)

        # Organize by problem
        train_problems = defaultdict(list)
        for item in train_data:
            train_problems[item['problem_id']].append(item)

        # Filter to only problems with both correct and incorrect
        train_problems = {
            pid: rollouts for pid, rollouts in train_problems.items()
            if any(r['is_correct'] for r in rollouts) and any(not r['is_correct'] for r in rollouts)
        }

        print(f"Training on {len(train_problems)} problems with both correct/incorrect rollouts")
        print(f"Total correct rollouts: {sum(r['is_correct'] for rolls in train_problems.values() for r in rolls)}")
        print(f"Total incorrect rollouts: {sum(not r['is_correct'] for rolls in train_problems.values() for r in rolls)}")

        # Create model
        input_dim = len(self.metric_keys)
        self.model = ConfidenceNet(input_dim, hidden_dim=self.hidden_dim, dropout=self.dropout)
        self.model.to(self.device)

        print(f"Model architecture:")
        print(f"  Input: {input_dim} features")
        print(f"  Hidden: [{self.hidden_dim}, {self.hidden_dim//2}]")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters())}")
        sys.stdout.flush()

        # Train
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_accuracy = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            total_loss = 0.0
            num_problems = 0

            for problem_id, rollouts in train_problems.items():
                correct_rollouts = [r for r in rollouts if r['is_correct']]
                incorrect_rollouts = [r for r in rollouts if not r['is_correct']]

                if not correct_rollouts or not incorrect_rollouts:
                    continue

                # Get features
                correct_features = torch.tensor(np.array([r['features'] for r in correct_rollouts])).to(self.device)
                incorrect_features = torch.tensor(np.array([r['features'] for r in incorrect_rollouts])).to(self.device)

                # Compute confidences
                correct_confidences = self.model(correct_features)
                incorrect_confidences = self.model(incorrect_features)

                # Max-margin objective
                min_correct = torch.min(correct_confidences)
                max_incorrect = torch.max(incorrect_confidences)

                problem_loss = torch.relu(max_incorrect - min_correct + self.margin)
                total_loss += problem_loss
                num_problems += 1

            # Average loss
            if num_problems > 0:
                loss = total_loss / num_problems
            else:
                loss = torch.tensor(0.0)

            loss.backward()
            optimizer.step()

            # Evaluate every N epochs
            if epoch % self.eval_interval == 0:
                # Evaluate on validation data if available, otherwise on training data
                if val_data:
                    accuracy = self._evaluate_accuracy(val_data)
                    metric_name = "Val Acc"
                else:
                    accuracy = self._evaluate_accuracy(train_data)
                    metric_name = "Train Acc"

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f"Epoch {epoch}/{self.num_epochs}: Loss={loss.item():.4f}, {metric_name}={accuracy:.2%}, Best={best_accuracy:.2%}")
                sys.stdout.flush()
                
                # Early stopping
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\n✓ Restored model to best checkpoint with accuracy: {best_accuracy:.2%}")
            sys.stdout.flush()

        self.is_trained = True

        return {
            'best_accuracy': best_accuracy,
            'final_epoch': self.num_epochs
        }

    def _evaluate_accuracy(self, data: List[Dict]) -> float:
        """Helper to evaluate accuracy on data."""
        self.model.eval()

        problems = defaultdict(list)
        for item in data:
            problems[item['problem_id']].append(item)

        correct_count = 0
        total_count = 0

        with torch.no_grad():
            for problem_id, rollouts in problems.items():
                features = torch.tensor(np.array([r['features'] for r in rollouts])).to(self.device)
                confidences = self.model(features).cpu().numpy()

                best_idx = np.argmax(confidences)
                if rollouts[best_idx]['is_correct']:
                    correct_count += 1
                total_count += 1

        return correct_count / total_count if total_count > 0 else 0

    def predict_confidence(self, data_item: Dict) -> float:
        """
        Predict confidence using trained model.

        Args:
            data_item: Dictionary with 'features' key

        Returns:
            Confidence score
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Method must be trained before prediction")

        # Need to ensure data_item has features
        if 'features' not in data_item:
            raise ValueError("Data item must have 'features' key. Call _load_metrics_data first.")

        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(data_item['features']).unsqueeze(0).to(self.device)
            confidence = self.model(features).cpu().item()

        return confidence

    def evaluate(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Override evaluate to load metrics first."""
        # Load metrics
        test_data = self._load_metrics_data(test_data)

        # Call parent evaluate
        return super().evaluate(test_data)

    def save(self, save_dir: Path) -> None:
        """Save model and metadata."""
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metric_keys': self.metric_keys,
            'input_dim': len(self.metric_keys)
        }, save_dir / 'model.pt')

        # Save metadata
        metadata = {
            'name': self.name,
            'hidden_dim': self.hidden_dim,
            'num_epochs': self.num_epochs,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'margin': self.margin,
            'seed': self.seed,
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

        self.hidden_dim = metadata['hidden_dim']
        self.num_epochs = metadata['num_epochs']
        self.lr = metadata['lr']
        self.weight_decay = metadata['weight_decay']
        self.margin = metadata['margin']
        self.seed = metadata.get('seed')
        self.is_trained = metadata['is_trained']

        # Load model
        checkpoint = torch.load(load_dir / 'model.pt', map_location=self.device)
        self.metric_keys = checkpoint['metric_keys']
        input_dim = checkpoint['input_dim']

        self.model = ConfidenceNet(input_dim, hidden_dim=self.hidden_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        print(f"\n✓ {self.name} loaded from {load_dir}")
