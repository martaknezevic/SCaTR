"""
Convex optimization method for learning logprob weights.

Based on optimize_logprob_weights.py
"""

import numpy as np
import cvxpy as cp
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from time import time
import sys

from methods.base_method import BaseMethod
from methods.method_registry import register_method

@register_method('convex')
class ConvexOptimizationMethod(BaseMethod):
    """
    Learn logprob weights using convex optimization.

    Uses a ranking loss approach with hinge loss to encourage correct
    responses to have higher confidence than incorrect ones.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Required parameters
        self.solver = config['solver']
        self.margin = config['margin']
        self.max_iters = config['max_iters']
        # Optional parameters
        self.tail_tokens = config.get('tail_tokens', None)
        self.weights = None
        self.num_top_logprobs = None

    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Train convex optimization model.

        Args:
            train_data: Training data
            val_data: Validation data (not used for convex optimization)
            output_dir: Directory to save model (optional)

        Returns:
            Training metrics
        """
        print(f"\n{'='*80}")
        print(f"{self.name}: Setting up optimization problem...")
        print(f"{'='*80}")
        sys.stdout.flush()

        # Organize by problem
        problems = defaultdict(list)
        for item in train_data:
            problems[item['problem_id']].append(item)

        print(f"Number of training problems: {len(problems)}")

        # Determine num_top_logprobs
        self.num_top_logprobs = train_data[0]['logprobs'].shape[1] if train_data else 10
        print(f"Number of top logprobs per token: {self.num_top_logprobs}")

        # Decision variable: weights for top_logprobs
        weights_var = cp.Variable(self.num_top_logprobs)

        # Constraints
        constraints = [
            weights_var >= 0,
            cp.sum(weights_var) == 1
        ]

        print(f"Constraints: weights >= 0, sum(weights) = 1")

        # Objective: minimize ranking loss
        losses = []
        pair_count = 0

        for problem_id, rollouts in problems.items():
            correct_rollouts = [r for r in rollouts if r['is_correct']]
            incorrect_rollouts = [r for r in rollouts if not r['is_correct']]

            if not correct_rollouts or not incorrect_rollouts:
                continue

            # Add pairwise ranking constraints
            for correct in correct_rollouts:
                for incorrect in incorrect_rollouts:
                    correct_lp = correct['logprobs']
                    incorrect_lp = incorrect['logprobs']

                    # Apply tail tokens selection if specified
                    if self.tail_tokens is not None and self.tail_tokens > 0:
                        correct_lp = correct_lp[-self.tail_tokens:]
                        incorrect_lp = incorrect_lp[-self.tail_tokens:]

                    # Average logprob per token
                    correct_avg_lp = cp.sum(correct_lp @ weights_var) / correct_lp.shape[0]
                    incorrect_avg_lp = cp.sum(incorrect_lp @ weights_var) / incorrect_lp.shape[0]

                    # Hinge loss
                    loss = cp.pos(self.margin - (incorrect_avg_lp - correct_avg_lp))
                    losses.append(loss)
                    pair_count += 1

        print(f"Number of (correct, incorrect) pairs: {pair_count}")
        print(f"Margin: {self.margin}")
        sys.stdout.flush()

        # Total loss
        total_loss = cp.sum(losses)

        # Formulate problem
        problem = cp.Problem(cp.Minimize(total_loss), constraints)

        print(f"\nOptimization problem created")
        print(f"  Variables: {problem.size_metrics.num_scalar_variables}")
        print(f"  Constraints: {problem.size_metrics.num_scalar_eq_constr + problem.size_metrics.num_scalar_leq_constr}")
        sys.stdout.flush()

        # Solve
        print(f"\n{self.name}: Solving optimization problem...")
        print(f"Solver: {self.solver}")
        sys.stdout.flush()

        solve_start = time()
        problem.solve(verbose=True, solver=getattr(cp, self.solver), max_iters=self.max_iters)
        solve_time = time() - solve_start

        print(f"\nSolving took: {solve_time:.2f}s")
        print(f"Optimization status: {problem.status}")
        print(f"Optimal value: {problem.value:.6f}")
        sys.stdout.flush()

        if weights_var.value is None:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        self.weights = weights_var.value
        self.is_trained = True

        print(f"\nLearned weights (sum={np.sum(self.weights):.6f}):")
        for i, w in enumerate(self.weights):
            print(f"  Top-{i+1} logprob: {w:.6f}")
        sys.stdout.flush()

        return {
            'solve_time': solve_time,
            'optimization_status': problem.status,
            'optimal_value': problem.value,
            'num_pairs': pair_count,
            'weights': self.weights.tolist()
        }

    def predict_confidence(self, data_item: Dict) -> float:
        """
        Predict confidence using learned weights.

        Args:
            data_item: Dictionary with 'logprobs' key

        Returns:
            Confidence score
        """
        if not self.is_trained or self.weights is None:
            raise ValueError("Method must be trained before prediction")

        logprobs = data_item['logprobs']

        if logprobs.size == 0:
            return 0.0

        # Select tokens
        if self.tail_tokens is not None and self.tail_tokens > 0:
            logprobs_to_use = logprobs[-self.tail_tokens:]
        else:
            logprobs_to_use = logprobs

        if logprobs_to_use.size == 0:
            return 0.0

        # Compute weighted average
        token_confidences = -np.dot(logprobs_to_use, self.weights)

        return np.mean(token_confidences)

    def save(self, save_dir: Path) -> None:
        """Save learned weights and metadata."""
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        np.save(save_dir / 'weights.npy', self.weights)

        # Save metadata
        metadata = {
            'name': self.name,
            'solver': self.solver,
            'margin': self.margin,
            'max_iters': self.max_iters,
            'tail_tokens': self.tail_tokens,
            'num_top_logprobs': self.num_top_logprobs,
            'is_trained': self.is_trained
        }
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ {self.name} saved to {save_dir}")

    def load(self, load_dir: Path) -> None:
        """Load learned weights and metadata."""
        # Load weights
        self.weights = np.load(load_dir / 'weights.npy')

        # Load metadata
        with open(load_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.solver = metadata['solver']
        self.margin = metadata['margin']
        self.max_iters = metadata['max_iters']
        self.tail_tokens = metadata['tail_tokens']
        self.num_top_logprobs = metadata['num_top_logprobs']
        self.is_trained = metadata['is_trained']

        print(f"\n✓ {self.name} loaded from {load_dir}")
