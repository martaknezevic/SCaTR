from methods.base_method import BaseMethod
from methods.method_registry import register_method
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys

class MethodA(BaseMethod):
    """Random baseline implementation"""
    
    def __init__(self, method_name: str, params: dict):
        """Initialize MethodA with specific parameters"""
        pass
    
    def prepare_data(self, train_data, test_data):
        """Prepare data for MethodA"""
        pass
    
    def train(self, train_data, output_dir: str):
        """Train model using MethodA approach"""
        pass
    
    def predict(self, test_data) -> list:
        """Generate predictions using MethodA"""
        pass
    
    def select_rollout(self, data_item: Dict) -> int:
        """Select rollout index for a given data item"""
        pass
    
    
@register_method('random')
class RandomMethod(BaseMethod):
    """
    Random selection baseline.

    For each problem, randomly selects a rollout.
    Since this is stochastic, we run multiple trials and report the mean.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_trials = config.get('num_trials', 100)
        self.is_trained = True  # No training needed

    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Random method doesn't require training."""
        print(f"\n{self.name}: No training required (random baseline)")
        return {'note': 'No training required for random baseline'}

    def predict_confidence(self, data_item: Dict) -> float:
        """Return random confidence score."""
        return np.random.random()

    def save(self, save_dir: Path) -> None:
        """Save method metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            'name': self.name,
            'num_trials': self.num_trials
        }
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, load_dir: Path) -> None:
        """Load method metadata."""
        with open(load_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        self.num_trials = metadata['num_trials']
        
    def evaluate(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Override evaluate to run multiple trials and report statistics.
        """
        import random

        print(f"\n{self.name}: Running {self.num_trials} random trials...")
        sys.stdout.flush()

        # Group by problem_id
        problems = defaultdict(list)
        for item in test_data:
            problems[item['problem_id']].append(item)

        trial_accuracies = []
        for trial in range(self.num_trials):
            correct_count = 0
            for problem_id, rollouts in sorted(problems.items(), key=lambda x: x[0]):  
                selected = random.choice(rollouts)
                if selected['is_correct']:
                    correct_count += 1

            accuracy = correct_count / len(problems)
            trial_accuracies.append(accuracy)

        mean_accuracy = np.mean(trial_accuracies)
        std_accuracy = np.std(trial_accuracies)

        print(f"{self.name} accuracy: {mean_accuracy:.2%} ± {std_accuracy:.2%}")
        sys.stdout.flush()

        return {
            'accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'correct_count': int(mean_accuracy * len(problems)),
            'total_count': len(problems),
            'num_trials': self.num_trials
        }


@register_method('uniform')
class UniformMethod(BaseMethod):
    """
    Uniform weights baseline.

    Computes confidence as the negative average of uniform-weighted top logprobs.
    This is equivalent to using the mean logprob as confidence.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tail_tokens = config.get('tail_tokens', None)
        self.is_trained = True  # No training needed

    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Uniform method doesn't require training."""
        print(f"\n{self.name}: No training required (uniform weights baseline)")
        return {'note': 'No training required for uniform weights baseline'}

    def predict_confidence(self, data_item: Dict) -> float:
        """
        Compute confidence as negative average of logprobs.

        Args:
            data_item: Dictionary with 'logprobs' key

        Returns:
            Confidence score
        """
        logprobs = data_item['logprobs']

        if logprobs.size == 0:
            return 0.0

        # Select tokens (tail or all)
        if self.tail_tokens is not None and self.tail_tokens > 0:
            logprobs_to_use = logprobs[-self.tail_tokens:]
        else:
            logprobs_to_use = logprobs

        if logprobs_to_use.size == 0:
            return 0.0

        # Uniform weights
        weights = np.ones(logprobs_to_use.shape[1]) / logprobs_to_use.shape[1]

        # Compute weighted average for each token (negative because logprobs are negative)
        token_confidences = -np.dot(logprobs_to_use, weights)

        # Average across tokens
        return np.mean(token_confidences)

    def save(self, save_dir: Path) -> None:
        """Save method metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            'name': self.name,
            'tail_tokens': self.tail_tokens
        }
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, load_dir: Path) -> None:
        """Load method metadata."""
        with open(load_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        self.tail_tokens = metadata['tail_tokens']
        
@register_method('oracle')
class OracleMethod(BaseMethod):
    """
    Oracle method that always selects the correct rollout if available.
    Used for upper-bound performance estimation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.is_trained = True  # No training needed

    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Oracle method doesn't require training."""
        print(f"\n{self.name}: No training required (oracle method)")
        return {'note': 'No training required for oracle method'}

    def predict_confidence(self, data_item: Dict) -> float:
        """Oracle confidence is always maximal."""
        return data_item.get('is_correct', 0.0)

    def save(self, save_dir: Path) -> None:
        """Save method metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            'name': self.name
        }
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, load_dir: Path) -> None:
        """Load method metadata."""
        pass  # No parameters to load
