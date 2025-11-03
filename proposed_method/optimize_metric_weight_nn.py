#!/usr/bin/env python3
"""
Neural Network approach for learning non-linear metric combinations.
Uses PyTorch to train a small feedforward network.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import sys

# Paths
TURN=7
PARENT_DIR = f"/home/ubuntu/cactts/big_results/qwen8b/MathArena_aime_2025/turn{TURN}"
METRICS_FILE = Path(f"{PARENT_DIR}/all_response_metrics.jsonl")

class ConfidenceNet(nn.Module):
    """Small neural network to learn confidence scores from metrics."""

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data() -> Tuple[List[Dict], List[str]]:
    """Load metrics from JSONL file."""
    print("Loading metrics from JSONL file...")
    sys.stdout.flush()

    all_data = []
    metric_keys = None

    with open(METRICS_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Extract numeric metrics on first iteration
            if metric_keys is None:
                # Exclude is_correct, rollout_idx, and extracted_answer (it's the predicted value, not a metric)
                excluded = ['rollout_idx', 'is_correct', 'extracted_answer', 'expected_answer']
                metric_keys = [k for k, v in data.items()
                             if isinstance(v, (int, float)) and k not in excluded]
                print(f"Found {len(metric_keys)} numeric metrics")
                sys.stdout.flush()

            # Extract features
            features = np.array([data[k] for k in metric_keys], dtype=np.float32)

            all_data.append({
                'problem_id': data['problem_id'],
                'rollout_idx': data['rollout_idx'],
                'features': features,
                'is_correct': data['is_correct']
            })

    print(f"Loaded {len(all_data)} rollouts")
    sys.stdout.flush()

    return all_data, metric_keys


def prepare_training_data(all_data: List[Dict]) -> Tuple:
    """
    Prepare training data for neural network.

    We'll use pairwise ranking: for each problem, create pairs of
    (correct, incorrect) and train network to rank correct higher.
    """
    # Organize by problem
    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    # Create pairwise training examples
    X_correct = []
    X_incorrect = []

    for problem_id, rollouts in problems.items():
        correct_rollouts = [r for r in rollouts if r['is_correct']]
        incorrect_rollouts = [r for r in rollouts if not r['is_correct']]

        if not correct_rollouts or not incorrect_rollouts:
            continue

        # Create all pairs
        for correct in correct_rollouts:
            for incorrect in incorrect_rollouts:
                X_correct.append(correct['features'])
                X_incorrect.append(incorrect['features'])

    X_correct = np.array(X_correct, dtype=np.float32)
    X_incorrect = np.array(X_incorrect, dtype=np.float32)

    print(f"\nTraining data: {len(X_correct)} pairs")
    sys.stdout.flush()

    return X_correct, X_incorrect, problems


def evaluate_model(model: nn.Module, all_data: List[Dict]) -> Dict:
    """
    Evaluate model performance.

    For each problem, compute confidence for all rollouts,
    then check if the rollout with highest confidence is correct.
    """
    model.eval()

    # Group by problem
    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    correct_count = 0
    total_count = 0
    problem_results = {}

    with torch.no_grad():
        for problem_id, rollouts in problems.items():
            # Compute confidence for all rollouts
            features = torch.tensor(np.array([r['features'] for r in rollouts]))
            confidences = model(features).numpy()

            # Find best rollout
            best_idx = np.argmax(confidences)
            best_rollout = rollouts[best_idx]

            is_correct = best_rollout['is_correct']
            if is_correct:
                correct_count += 1
            total_count += 1

            problem_results[problem_id] = {
                'best_rollout_idx': best_rollout['rollout_idx'],
                'best_confidence': float(confidences[best_idx]),
                'is_correct': is_correct,
                'num_rollouts': len(rollouts)
            }

    return {
        'accuracy': correct_count / total_count if total_count > 0 else 0,
        'correct_count': correct_count,
        'total_count': total_count,
        'problem_results': problem_results
    }


def compute_baseline_performance(all_data: List[Dict]) -> Dict:
    """Compute baseline with uniform weights (mean of features)."""
    print("\nComputing uniform weights baseline...")
    sys.stdout.flush()

    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    correct_count = 0
    total_count = 0
    total_correct_rollouts = 0
    total_rollouts = 0

    for problem_id, rollouts in problems.items():
        total_rollouts += len(rollouts)
        total_correct_rollouts += sum(1 for r in rollouts if r['is_correct'])

        # Use mean of features as confidence
        confidences = [np.mean(r['features']) for r in rollouts]
        best_idx = np.argmax(confidences)

        if rollouts[best_idx]['is_correct']:
            correct_count += 1
        total_count += 1

    print(f"Evaluated {total_count} problems")
    print(f"Total rollouts: {total_rollouts}, Correct rollouts: {total_correct_rollouts} ({total_correct_rollouts/total_rollouts:.2%})")
    sys.stdout.flush()

    return {
        'accuracy': correct_count / total_count if total_count > 0 else 0,
        'correct_count': correct_count,
        'total_count': total_count,
        'total_correct_rollouts': total_correct_rollouts,
        'total_rollouts': total_rollouts
    }


def compute_random_baseline(all_data: List[Dict], num_trials: int = 100) -> Dict:
    """Compute random selection baseline."""
    import random

    print(f"\nComputing random selection baseline ({num_trials} trials)...")
    sys.stdout.flush()

    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    trial_accuracies = []
    for _ in range(num_trials):
        correct_count = 0
        for problem_id, rollouts in problems.items():
            if random.choice(rollouts)['is_correct']:
                correct_count += 1
        trial_accuracies.append(correct_count / len(problems))

    mean_accuracy = np.mean(trial_accuracies)
    std_accuracy = np.std(trial_accuracies)

    print(f"Random selection accuracy: {mean_accuracy:.2%} ± {std_accuracy:.2%}")
    sys.stdout.flush()

    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'num_trials': num_trials
    }


def compute_single_metric_performance(all_data: List[Dict], metric_idx: int,
                                     metric_name: str, reverse: bool = False) -> Dict:
    """Evaluate performance using a single metric."""
    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    correct_count = 0
    total_count = 0

    for problem_id, rollouts in problems.items():
        metric_values = []
        for rollout in rollouts:
            value = rollout['features'][metric_idx]
            metric_values.append({
                'rollout_idx': rollout['rollout_idx'],
                'value': value,
                'is_correct': rollout['is_correct']
            })

        # Find rollout with best metric value
        best_rollout = max(metric_values, key=lambda x: -x['value'] if reverse else x['value'])

        if best_rollout['is_correct']:
            correct_count += 1
        total_count += 1

    return {
        'metric_name': metric_name,
        'reverse': reverse,
        'accuracy': correct_count / total_count if total_count > 0 else 0,
        'correct_count': correct_count,
        'total_count': total_count
    }


def find_best_individual_metrics(all_data: List[Dict], metric_keys: List[str],
                                 top_k: int = 5) -> List[Dict]:
    """Find the top-k best performing individual metrics."""
    print(f"\nEvaluating individual metrics (finding best {top_k})...")
    sys.stdout.flush()

    all_results = []

    for idx, metric_name in enumerate(metric_keys):
        for reverse in [False, True]:
            result = compute_single_metric_performance(all_data, idx, metric_name, reverse)
            all_results.append(result)

    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    top_results = all_results[:top_k]

    print(f"\nTop {top_k} individual metrics:")
    for i, result in enumerate(top_results):
        direction = "↓ (lower better)" if result['reverse'] else "↑ (higher better)"
        print(f"  {i+1}. {result['metric_name']} {direction}: {result['accuracy']:.2%} ({result['correct_count']}/{result['total_count']})")
    sys.stdout.flush()

    return top_results


def train_model_maxmargin(model: nn.Module, training_problems: Dict,
                          all_data: List[Dict], num_epochs: int = 1000, lr: float = 0.0005) -> List[float]:
    """
    Train the model using max-margin objective.

    For each problem, we want:
        min(correct_confidences) > max(incorrect_confidences) + margin

    Objective: Maximize sum over problems of [min(correct) - max(incorrect)]
    Equivalently: Minimize -[min(correct) - max(incorrect)]
    With hinge loss: Minimize max(0, max(incorrect) - min(correct) + margin)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    desired_margin = 1.0

    train_losses = []
    best_accuracy = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        num_problems = 0

        # For each problem with both correct and incorrect
        for problem_id, rollouts in training_problems.items():
            correct_rollouts = [r for r in rollouts if r['is_correct']]
            incorrect_rollouts = [r for r in rollouts if not r['is_correct']]

            if not correct_rollouts or not incorrect_rollouts:
                continue

            # Get features
            correct_features = torch.tensor(np.array([r['features'] for r in correct_rollouts]))
            incorrect_features = torch.tensor(np.array([r['features'] for r in incorrect_rollouts]))

            # Compute confidences
            correct_confidences = model(correct_features)
            incorrect_confidences = model(incorrect_features)

            # Max-margin objective: min(correct) should beat max(incorrect) by margin
            min_correct = torch.min(correct_confidences)
            max_incorrect = torch.max(incorrect_confidences)

            # Hinge loss: max(0, max(incorrect) - min(correct) + margin)
            problem_loss = torch.relu(max_incorrect - min_correct + desired_margin)
            total_loss += problem_loss
            num_problems += 1

        # Average loss over problems
        if num_problems > 0:
            loss = total_loss / num_problems
        else:
            loss = torch.tensor(0.0)

        # Backprop
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Evaluate every 100 epochs
        if (epoch + 1) % 100 == 0:
            eval_result = evaluate_model(model, all_data)
            accuracy = eval_result['accuracy']

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = model.state_dict().copy()

            print(f"Epoch {epoch+1}/{num_epochs}: Loss={loss.item():.4f}, "
                  f"Accuracy={accuracy:.2%} ({eval_result['correct_count']}/{eval_result['total_count']}), "
                  f"Best={best_accuracy:.2%}")
            sys.stdout.flush()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n✓ Restored model to best checkpoint with accuracy: {best_accuracy:.2%}")
        sys.stdout.flush()

    return train_losses, best_accuracy


def main():
    print("=" * 80)
    print("Neural Network Metric Combination for AIME 2025")
    print("=" * 80)

    # Load data
    print("\nStep 1: Loading data...")
    print("=" * 80)
    all_data, metric_keys = load_data()

    # Baselines
    print("\n" + "=" * 80)
    print("Step 2: Baseline Performance")
    print("=" * 80)

    random_baseline = compute_random_baseline(all_data)
    baseline = compute_baseline_performance(all_data)
    print(f"\n✓ Uniform Weights Baseline: {baseline['accuracy']:.2%} ({baseline['correct_count']}/{baseline['total_count']})")

    best_metrics = find_best_individual_metrics(all_data, metric_keys, top_k=5)

    # Use ALL features (no feature selection)
    print("\n" + "=" * 80)
    print("Step 3: Using all features...")
    print("=" * 80)
    print(f"Using all {len(metric_keys)} metrics (no feature selection)")
    sys.stdout.flush()

    # Organize data by problem for max-margin training
    print("\n" + "=" * 80)
    print("Step 4: Organizing data by problem...")
    print("=" * 80)

    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    # Filter to only problems with both correct and incorrect
    training_problems = {
        pid: rollouts for pid, rollouts in problems.items()
        if any(r['is_correct'] for r in rollouts) and any(not r['is_correct'] for r in rollouts)
    }

    print(f"Total problems: {len(problems)}")
    print(f"Problems with both correct/incorrect (for training): {len(training_problems)}")
    sys.stdout.flush()

    # Create model
    print("\n" + "=" * 80)
    print("Step 5: Training neural network...")
    print("=" * 80)

    input_dim = len(metric_keys)
    hidden_dim = 32  # Larger network for more features
    model = ConfidenceNet(input_dim, hidden_dim=hidden_dim)

    print(f"Model architecture:")
    print(f"  Input: {input_dim} features")
    print(f"  Hidden: [{hidden_dim}, {hidden_dim//2}]")
    print(f"  Output: 1 (confidence score)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    sys.stdout.flush()

    # Train with max-margin objective
    train_losses, best_training_accuracy = train_model_maxmargin(model, training_problems, all_data,
                                                                  num_epochs=3000, lr=0.001)

    # Final evaluation (model has been restored to best checkpoint)
    print("\n" + "=" * 80)
    print("Step 6: Final Evaluation (Best Model)")
    print("=" * 80)

    nn_result = evaluate_model(model, all_data)

    print(f"\nBest accuracy during training: {best_training_accuracy:.2%}")
    print(f"Final evaluation accuracy: {nn_result['accuracy']:.2%}")

    if abs(best_training_accuracy - nn_result['accuracy']) > 0.001:
        print(f"⚠ Warning: Evaluation accuracy differs from best training accuracy!")
        print(f"  This may indicate evaluation inconsistency.")
    else:
        print(f"✓ Model successfully restored to best checkpoint")

    print(f"\n{'Metric':<25} {'Random':<20} {'Baseline':<20} {'Best-1':<20} {'Neural Net':<20}")
    print("-" * 105)
    print(f"{'Accuracy':<25} {random_baseline['mean_accuracy']:.2%}±{random_baseline['std_accuracy']:.2%}          "
          f"{baseline['accuracy']:.2%} ({baseline['correct_count']}/{baseline['total_count']})       "
          f"{best_metrics[0]['accuracy']:.2%} ({best_metrics[0]['correct_count']}/{best_metrics[0]['total_count']})       "
          f"{nn_result['accuracy']:.2%} ({nn_result['correct_count']}/{nn_result['total_count']})")

    # Save model
    model_file = Path("/home/ubuntu/cactts/learned_metric_nn.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'metric_keys': metric_keys,
        'input_dim': input_dim
    }, model_file)
    print(f"\n✓ Model saved to: {model_file}")

    # Save results
    results_file = Path("/home/ubuntu/cactts/metric_optimization_results_nn.json")
    results = {
        'random_baseline': {
            'mean_accuracy': random_baseline['mean_accuracy'],
            'std_accuracy': random_baseline['std_accuracy'],
            'num_trials': random_baseline['num_trials']
        },
        'baseline': {
            'accuracy': baseline['accuracy'],
            'correct_count': baseline['correct_count'],
            'total_count': baseline['total_count']
        },
        'best_individual_metric': {
            'name': best_metrics[0]['metric_name'],
            'reverse': best_metrics[0]['reverse'],
            'accuracy': best_metrics[0]['accuracy'],
            'correct_count': best_metrics[0]['correct_count'],
            'total_count': best_metrics[0]['total_count']
        },
        'neural_network': {
            'accuracy': nn_result['accuracy'],
            'correct_count': nn_result['correct_count'],
            'total_count': nn_result['total_count'],
            'problem_results': nn_result['problem_results'],
            'best_training_accuracy': best_training_accuracy,
            'note': 'Model was restored to best checkpoint from training'
        },
        'model_config': {
            'input_dim': input_dim,
            'hidden_dims': [hidden_dim, hidden_dim//2],
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'metric_names': metric_keys
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
