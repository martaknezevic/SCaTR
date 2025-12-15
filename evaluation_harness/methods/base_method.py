from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import torch
import random

class BaseMethod(ABC):
    """Abstract base class for all evaluation methods"""
    
    def __init__(self, config: dict):
        """
        Initialize method
        
        Args:
            config: Method configuration dictionary including all parameters
        """
        self.config = config
        self.name = config.get('method_name', self.__class__.__name__)
        self.seed = config.get('seed', 42)
        self.is_trained = False
        self.model_path = None
    

    def prepare_data(self, train_data: List[Dict], val_data: List[Dict] = None, test_data: List[Dict] = None):
        """
        Prepare data for this specific method.
        
        This method can be overridden to perform method-specific data preprocessing,
        feature extraction, or data formatting. By default, returns data unchanged.
        
        Args:
            train_data: Training dataset (list of dictionaries)
            val_data: Validation dataset (optional)
            test_data: Test dataset (optional)
            
        Returns:
            Tuple of (prepared_train_data, prepared_val_data, prepared_test_data)
        """
        # Default implementation: no preprocessing needed
        return train_data, val_data, test_data
    
    @abstractmethod
    def train(self, train_data: List[Dict], val_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Train model if required.
        
        This method should train the model on training data and optionally validate on val_data.
        If the method doesn't require training (e.g., baseline heuristics), 
        this can be a no-op that returns immediately.
        
        Args:
            train_data: Prepared training data
            val_data: Prepared validation data (optional)
            
        Returns:
            Dictionary with training metrics (e.g., {'train_loss': 0.5, 'val_accuracy': 0.8})
        """
        pass
    
    @abstractmethod
    def predict_confidence(self, rollout: Dict) -> float:
        """
        Predict confidence score for a single rollout.
        
        This is the core method that each subclass must implement.
        It should return a confidence score (higher = more confident) for the given rollout.
        
        Args:
            rollout: Single rollout dictionary with fields like 'logprobs', 'answer', etc.
            
        Returns:
            Confidence score (float)
        """
        pass
    
    def predict_batch(self, rollouts: List[Dict]) -> List[float]:
        """
        Predict confidence scores for a batch of rollouts.
        
        By default, this calls predict_confidence for each rollout.
        Subclasses can override this for more efficient batch processing.
        
        Args:
            rollouts: List of rollout dictionaries
            
        Returns:
            List of confidence scores
        """
        return [self.predict_confidence(rollout) for rollout in rollouts]
    
    def save(self, output_dir: str) -> str:
        """
        Save trained model to disk.
        
        Args:
            output_dir: Directory to save model
            
        Returns:
            Path to saved model file
        """
        # Default implementation: no model to save
        return None
    
    def load_model(self, model_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
        """
        # Default implementation: no model to load
        pass
    
    def format_output(self, test_data: List[Dict], predictions: List[float] = None) -> List[Dict]:
        """
        Format predictions into standard JSON format for evaluation.
        
        This method creates a list of dictionaries in the standard format:
        {
            "problem_id": "...",
            "rollout_idx": 123,
            "confidence": 0.95,
            "answer": "...",
            "is_correct": true,
            "prompt": "..."  # if available
        }
        
        For each problem, the rollout with highest confidence is marked as chosen.
        
        Args:
            test_data: List of test rollouts
            predictions: List of confidence scores (optional, will compute if not provided)
            
        Returns:
            List of formatted prediction dictionaries
        """
        from collections import defaultdict
        
        # Group by problem_id
        problems = defaultdict(list)
        for i, rollout in enumerate(test_data):
            conf = predictions[i] if predictions else self.predict_confidence(rollout)
            problems[rollout['problem_id']].append({
                'rollout': rollout,
                'confidence': conf,
                'index': i
            })
        
        # For each problem, select the rollout with highest confidence
        output = []
        for problem_id, candidates in problems.items():
            # Find best rollout
            best = max(candidates, key=lambda x: x['confidence'])
            rollout = best['rollout']
            
            # Format output
            result = {
                'problem_id': problem_id,
                'rollout_idx': rollout.get('rollout_idx', best['index']),
                'confidence': best['confidence'],
                'answer': rollout.get('answer', ''),
                'is_correct': rollout.get('is_correct', False),
            }
            
            # Add prompt if available
            if 'prompt' in rollout:
                result['prompt'] = rollout['prompt']
            
            output.append(result)
        
        return output
    
    def save_predictions(self, predictions: List[Dict], output_path: str):
        """
        Save predictions to JSONL file.
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Path to save predictions
        """
        import json
        from pathlib import Path
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
    
    def get_method_id(self) -> str:
        """
        Get unique identifier for this method instance.
        
        Returns:
            String identifier based on method name and key parameters
        """
        # Include key params in ID for uniqueness
        param_keys = ['learning_rate', 'hidden_size', 'num_layers', 'alpha', 'beta']
        param_str_parts = []
        
        for key in param_keys:
            if key in self.config:
                value = self.config[key]
                if isinstance(value, float):
                    param_str_parts.append(f"{key}={value:.4f}")
                else:
                    param_str_parts.append(f"{key}={value}")
        
        if param_str_parts:
            return f"{self.name}_{'_'.join(param_str_parts)}"
        return self.name
    
    def evaluate(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate the method on test data.

        This is a common implementation that works for all methods.
        For each problem, select the rollout with highest confidence
        and check if it's correct.

        Args:
            test_data: List of test data items

        Returns:
            Dictionary with evaluation metrics
        """
        from collections import defaultdict

        # Group by problem_id
        problems = defaultdict(list)
        for item in test_data:
            problems[item['problem_id']].append(item)

        correct_count = 0
        total_count = 0
        problem_results = {}

        for problem_id, rollouts in problems.items():
            # Compute confidence for each rollout
            rollout_scores = []
            for rollout in rollouts:
                confidence = self.predict_confidence(rollout)
                rollout_scores.append({
                    'rollout_idx': rollout['rollout_idx'],
                    'confidence': confidence,
                    'is_correct': rollout['is_correct']
                })

            # Find rollout with highest confidence
            best_rollout = max(rollout_scores, key=lambda x: x['confidence'])

            is_correct = best_rollout['is_correct']
            if is_correct:
                correct_count += 1
            total_count += 1

            problem_results[problem_id] = {
                'best_rollout_idx': best_rollout['rollout_idx'],
                'best_confidence': best_rollout['confidence'],
                'is_correct': is_correct,
                'num_rollouts': len(rollouts),
                'num_correct_rollouts': sum(1 for r in rollouts if r['is_correct'])
            }

        return {
            'accuracy': correct_count / total_count if total_count > 0 else 0,
            'correct_count': correct_count,
            'total_count': total_count,
            'problem_results': problem_results
        }
    
    def set_seed(self, seed: int = None):
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed to use. If None, uses self.seed
        """
        seed = seed if seed is not None else self.seed
        
        if seed is not None:
            
            random.seed(seed)
            np.random.seed(seed)
            
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    # Make cudnn deterministic
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except ImportError:
                # torch not available, skip torch seeding
                pass
            
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the method for logging purposes.

        Returns:
            Dictionary with method metadata
        """
        return {
            'name': self.name,
            'config': self.config,
            'is_trained': self.is_trained
        }