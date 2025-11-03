"""
Optimized version: Learn logprob weights to maximize answer correctness.

This version:
1. Uses parallel processing for faster data loading
2. Works on a subset first for quick testing
3. Includes detailed progress tracking
4. Verbose optimization output
"""

import pickle
import json
import numpy as np
import cvxpy as cp
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from time import time
import multiprocessing as mp

# Default paths (can be overridden by command line arguments)
DEFAULT_TURN = "turn1"

def load_metrics(metrics_file: Path) -> Dict[Tuple[str, int], Dict]:
    """Load metrics JSONL file."""
    print("Loading metrics file...")
    sys.stdout.flush()
    metrics = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            key = (data['problem_id'], data['rollout_idx'])
            metrics[key] = data
    print(f"Loaded metrics for {len(metrics)} rollouts")
    sys.stdout.flush()
    return metrics

def extract_top_logprobs_from_choice(choice) -> np.ndarray:
    """
    Extract top logprobs for all tokens in a choice.

    Returns:
        np.ndarray of shape (num_tokens, num_top_logprobs)
    """
    # Handle both dict and object attribute access
    if isinstance(choice, dict):
        logprobs = choice.get('logprobs')
        if not logprobs or not logprobs.get('content'):
            return np.array([])
        content = logprobs['content']
    else:
        if not choice.logprobs or not choice.logprobs.content:
            return np.array([])
        content = choice.logprobs.content

    logprobs_list = []
    for token_logprob in content:
        # Handle both dict and object attribute access for top_logprobs
        if isinstance(token_logprob, dict):
            top_lps = [lp['logprob'] for lp in token_logprob['top_logprobs']]
        else:
            top_lps = [lp.logprob for lp in token_logprob.top_logprobs]
        logprobs_list.append(top_lps)

    return np.array(logprobs_list)

def process_single_choice(args: Tuple) -> Tuple[int, Dict]:
    """
    Process a single choice (rollout) and extract logprobs.

    Args:
        args: Tuple of (rollout_idx, choice, problem_id, metrics)

    Returns:
        Tuple of (rollout_idx, result_dict or None)
    """
    rollout_idx, choice, problem_id, metrics = args
    key = (problem_id, rollout_idx)

    if key not in metrics:
        return rollout_idx, None

    # Extract logprobs - this is the slow part
    logprobs = extract_top_logprobs_from_choice(choice)

    if logprobs.size == 0:
        return rollout_idx, None

    result = {
        'problem_id': problem_id,
        'rollout_idx': rollout_idx,
        'logprobs': logprobs,
        'is_correct': metrics[key]['is_correct']
    }

    return rollout_idx, result

def process_single_pickle(args: Tuple[Path, Dict, int]) -> Tuple[str, List[Dict], float]:
    """
    Process a single pickle file with nested parallelism for choices.

    Args:
        args: Tuple of (pkl_file, metrics, num_choice_workers)

    Returns:
        Tuple of (problem_id, results, processing_time)
    """
    pkl_file, metrics, num_choice_workers = args
    start_time = time()
    problem_id = pkl_file.stem.replace("_choices", "")

    with open(pkl_file, 'rb') as f:
        choices = pickle.load(f)

    # Prepare arguments for parallel choice processing
    choice_args = [
        (rollout_idx, choice, problem_id, metrics)
        for rollout_idx, choice in enumerate(choices)
    ]

    # Process choices in parallel using ThreadPoolExecutor
    # (threads are fine here since we're mostly doing numpy operations)
    results_dict = {}
    with ThreadPoolExecutor(max_workers=num_choice_workers) as executor:
        futures = {executor.submit(process_single_choice, arg): arg[0] for arg in choice_args}

        for future in as_completed(futures):
            rollout_idx, result = future.result()
            if result is not None:
                results_dict[rollout_idx] = result

    # Sort results by rollout_idx to maintain order
    results = [results_dict[idx] for idx in sorted(results_dict.keys())]

    process_time = time() - start_time
    return problem_id, results, process_time

def load_data_parallel(choices_dir: Path,
                       metrics_file: Path,
                       turn: str,
                       num_problems: int = None,
                       num_workers: int = 32,
                       num_choice_workers: int = 8,
                       use_cache: bool = True,
                       subsample_size: int = None) -> Tuple[List[Dict], Dict]:
    """
    Load pickle files in parallel with nested parallelism for choices.
    Uses caching to avoid reprocessing.

    Args:
        choices_dir: Directory containing pickle files
        metrics_file: Path to metrics JSONL file
        turn: Turn name (e.g., "turn1")
        num_problems: If specified, only load first N problems (for testing)
        num_workers: Number of parallel workers for pickle files (default: 32)
        num_choice_workers: Number of parallel workers per pickle file for choices (default: 8)
        use_cache: If True, save/load processed data from cache (default: True)
        subsample_size: If specified, randomly subsample this many rollouts per problem (default: None = use all)
    """
    # Define cache file path
    cache_suffix = f"_{num_problems}prob" if num_problems else "_full"
    if subsample_size:
        cache_suffix += f"_sub{subsample_size}"
    cache_file = Path(f"/home/ubuntu/cactts/proposed_method/results/{turn}/processed_data_cache{cache_suffix}.pkl")

    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"\n✓ Found cached data at {cache_file}")
        print("Loading from cache...")
        sys.stdout.flush()
        start = time()
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        load_time = time() - start
        print(f"✓ Loaded {len(cache_data['all_data'])} rollouts from cache in {load_time:.2f}s")
        sys.stdout.flush()
        return cache_data['all_data'], cache_data['metrics']

    # If no cache, process data
    print("\nNo cache found, processing data from scratch...")
    sys.stdout.flush()

    start = time()
    metrics = load_metrics(metrics_file)

    # Get all pickle files
    pickle_files = sorted(choices_dir.glob("aime_2025_*_choices.pkl"))

    if num_problems:
        pickle_files = pickle_files[:num_problems]
        print(f"\nProcessing subset: first {num_problems} problems")

    print(f"Found {len(pickle_files)} pickle files to process")
    print(f"Using {num_workers} parallel file workers, {num_choice_workers} choice workers per file")
    sys.stdout.flush()

    all_data = []

    # Prepare arguments for parallel processing
    args_list = [(pkl_file, metrics, num_choice_workers) for pkl_file in pickle_files]

    # Process in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_pickle, args): args[0]
                          for args in args_list}

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            pkl_file = future_to_file[future]
            try:
                problem_id, results, process_time = future.result()
                all_data.extend(results)
                completed += 1
                print(f"[{completed}/{len(pickle_files)}] ✓ {problem_id}: {len(results)} rollouts ({process_time:.2f}s)")
                sys.stdout.flush()
            except Exception as e:
                print(f"[{completed}/{len(pickle_files)}] ✗ {pkl_file.stem}: Error - {e}")
                sys.stdout.flush()

    total_time = time() - start
    print(f"\nTotal loading time: {total_time:.2f}s")
    print(f"Loaded {len(all_data)} valid rollouts")
    sys.stdout.flush()

    # Apply subsampling if requested
    if subsample_size is not None:
        import random
        random.seed(42)  # For reproducibility

        print(f"\nSubsampling {subsample_size} rollouts per problem...")
        sys.stdout.flush()

        # Group by problem_id
        problems = defaultdict(list)
        for item in all_data:
            problems[item['problem_id']].append(item)

        # Subsample from each problem
        subsampled_data = []
        for problem_id, rollouts in problems.items():
            if len(rollouts) <= subsample_size:
                subsampled_data.extend(rollouts)
            else:
                subsampled_data.extend(random.sample(rollouts, subsample_size))

        print(f"After subsampling: {len(subsampled_data)} rollouts (from {len(all_data)})")
        sys.stdout.flush()
        all_data = subsampled_data

    # Save to cache
    if use_cache:
        print(f"\nSaving processed data to cache: {cache_file}")
        sys.stdout.flush()
        with open(cache_file, 'wb') as f:
            pickle.dump({'all_data': all_data, 'metrics': metrics}, f)
        print("✓ Cache saved")
        sys.stdout.flush()

    return all_data, metrics

def compute_token_confidence(logprobs: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Compute per-token confidence as negative weighted average of top_logprobs.

    Args:
        logprobs: Array of shape (num_tokens, num_top_logprobs)
        weights: Array of shape (num_top_logprobs,). If None, uniform weights.

    Returns:
        Average token confidence across all tokens
    """
    if logprobs.size == 0:
        return 0.0

    if weights is None:
        # Uniform weights (baseline)
        weights = np.ones(logprobs.shape[1]) / logprobs.shape[1]

    # Compute weighted average for each token
    # logprobs are negative, so we negate them for confidence
    token_confidences = -np.dot(logprobs, weights)

    # Average across all tokens
    return np.mean(token_confidences)

def compute_random_baseline(all_data: List[Dict], num_trials: int = 100) -> Dict:
    """
    Compute random selection baseline.

    For each problem, randomly select a rollout and check if correct.
    Average over multiple trials to get expected accuracy.
    """
    import random

    print(f"\nComputing random selection baseline ({num_trials} trials)...")
    sys.stdout.flush()

    # Group rollouts by problem_id
    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    # Run multiple trials
    trial_accuracies = []
    for trial in range(num_trials):
        correct_count = 0
        for problem_id, rollouts in problems.items():
            # Randomly select a rollout
            selected = random.choice(rollouts)
            if selected['is_correct']:
                correct_count += 1

        accuracy = correct_count / len(problems)
        trial_accuracies.append(accuracy)

    mean_accuracy = np.mean(trial_accuracies)
    std_accuracy = np.std(trial_accuracies)

    print(f"Random selection accuracy: {mean_accuracy:.2%} ± {std_accuracy:.2%}")
    sys.stdout.flush()

    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'num_trials': num_trials
    }

def compute_baseline_performance(all_data: List[Dict]) -> Dict:
    """
    Compute baseline performance with uniform weights.

    For each problem_id, compute confidence for all 64 rollouts,
    then check if the rollout with highest confidence is correct.
    """
    print("\nComputing baseline performance...")
    sys.stdout.flush()

    # Group rollouts by problem_id
    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    # For each problem, compute confidence for all rollouts
    correct_count = 0
    total_count = 0
    total_correct_rollouts = 0
    total_rollouts = 0
    problem_results = {}

    for problem_id, rollouts in problems.items():
        # Count how many rollouts are correct for this problem
        num_correct_rollouts = sum(1 for r in rollouts if r['is_correct'])
        total_correct_rollouts += num_correct_rollouts
        total_rollouts += len(rollouts)

        # Compute confidence for each rollout in this problem
        rollout_confidences = []
        for rollout in rollouts:
            confidence = compute_token_confidence(rollout['logprobs'])
            rollout_confidences.append({
                'rollout_idx': rollout['rollout_idx'],
                'confidence': confidence,
                'is_correct': rollout['is_correct']
            })

        # Find rollout with highest confidence
        best_rollout = max(rollout_confidences, key=lambda x: x['confidence'])

        # Check if the highest confidence rollout is correct
        is_correct = best_rollout['is_correct']
        if is_correct:
            correct_count += 1
        total_count += 1

        problem_results[problem_id] = {
            'best_rollout_idx': best_rollout['rollout_idx'],
            'best_confidence': best_rollout['confidence'],
            'is_correct': is_correct,
            'num_rollouts': len(rollouts),
            'num_correct_rollouts': num_correct_rollouts
        }

    print(f"Evaluated {total_count} problems")
    print(f"Total rollouts: {total_rollouts}, Correct rollouts: {total_correct_rollouts} ({total_correct_rollouts/total_rollouts:.2%})")
    sys.stdout.flush()

    return {
        'accuracy': correct_count / total_count if total_count > 0 else 0,
        'correct_count': correct_count,
        'total_count': total_count,
        'total_correct_rollouts': total_correct_rollouts,
        'total_rollouts': total_rollouts,
        'problem_results': problem_results
    }

def setup_optimization(all_data: List[Dict], num_top_logprobs: int = 10) -> Tuple:
    """
    Set up convex optimization problem to learn weights.

    We use a ranking loss approach:
    For each problem, encourage correct responses to have higher confidence than incorrect ones.
    """
    print("\nSetting up optimization problem...")
    sys.stdout.flush()

    # Organize data by problem
    problems = defaultdict(list)
    for item in all_data:
        problems[item['problem_id']].append(item)

    print(f"Number of problems: {len(problems)}")
    sys.stdout.flush()

    # Decision variable: weights for top_logprobs
    weights = cp.Variable(num_top_logprobs)

    # Constraints: weights should be non-negative and sum to 1
    constraints = [
        weights >= 0,
        cp.sum(weights) == 1
    ]

    print("Constraints: weights >= 0, sum(weights) = 1")
    sys.stdout.flush()

    # Objective: minimize ranking loss
    losses = []
    margin = 0.1  # Margin for ranking loss
    pair_count = 0

    for problem_id, rollouts in problems.items():
        correct_rollouts = [r for r in rollouts if r['is_correct']]
        incorrect_rollouts = [r for r in rollouts if not r['is_correct']]

        if not correct_rollouts or not incorrect_rollouts:
            continue

        # Add pairwise ranking constraints
        for correct in correct_rollouts:
            for incorrect in incorrect_rollouts:
                # Confidence = -average(logprobs @ weights)
                # We want correct_conf > incorrect_conf
                # -avg(correct_lp @ w) > -avg(incorrect_lp @ w)
                # avg(incorrect_lp @ w) > avg(correct_lp @ w)

                correct_lp = correct['logprobs']
                incorrect_lp = incorrect['logprobs']

                # Average logprob per token
                correct_avg_lp = cp.sum(correct_lp @ weights) / correct_lp.shape[0]
                incorrect_avg_lp = cp.sum(incorrect_lp @ weights) / incorrect_lp.shape[0]

                # Hinge loss: max(0, margin - (incorrect - correct))
                loss = cp.pos(margin - (incorrect_avg_lp - correct_avg_lp))
                losses.append(loss)
                pair_count += 1

    print(f"Number of (correct, incorrect) pairs: {pair_count}")
    print(f"Margin: {margin}")
    sys.stdout.flush()

    # Total loss
    total_loss = cp.sum(losses)

    # Formulate problem
    problem = cp.Problem(cp.Minimize(total_loss), constraints)

    print(f"\nOptimization problem created:")
    print(f"  Variables: {problem.size_metrics.num_scalar_variables}")
    print(f"  Constraints: {problem.size_metrics.num_scalar_eq_constr + problem.size_metrics.num_scalar_leq_constr}")
    sys.stdout.flush()

    return problem, weights, problems

def evaluate_weights(all_data: List[Dict], weights: np.ndarray) -> Dict:
    """Evaluate performance with learned weights."""
    print("\nEvaluating learned weights...")
    sys.stdout.flush()

    problem_accuracies = defaultdict(list)

    for item in all_data:
        confidence = compute_token_confidence(item['logprobs'], weights)
        problem_accuracies[item['problem_id']].append({
            'confidence': confidence,
            'is_correct': item['is_correct'],
            'rollout_idx': item['rollout_idx']
        })

    # For each problem, check if highest confidence is correct
    correct_count = 0
    total_count = 0
    problem_results = {}

    for problem_id, rollouts in problem_accuracies.items():
        max_conf_rollout = max(rollouts, key=lambda x: x['confidence'])
        is_correct = max_conf_rollout['is_correct']

        if is_correct:
            correct_count += 1
        total_count += 1

        problem_results[problem_id] = {
            'best_rollout_idx': max_conf_rollout['rollout_idx'],
            'best_confidence': max_conf_rollout['confidence'],
            'is_correct': is_correct,
            'num_rollouts': len(rollouts)
        }

    return {
        'accuracy': correct_count / total_count if total_count > 0 else 0,
        'correct_count': correct_count,
        'total_count': total_count,
        'problem_results': problem_results
    }

def main():
    # Parse command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-problems', type=int, default=None,
                       help='Number of problems to process (default: all)')
    parser.add_argument('--num-workers', type=int, default=64,
                       help='Number of parallel workers for pickle files (default: 32)')
    parser.add_argument('--num-choice-workers', type=int, default=8,
                       help='Number of parallel workers per pickle file for choices (default: 8)')
    parser.add_argument('--solver', type=str, default='SCS',
                       help='CVXPY solver to use (SCS, ECOS, OSQP, etc.)')
    parser.add_argument('--subsample-size', type=int, default=None,
                       help='Number of rollouts to randomly subsample per problem (default: None = use all 64)')
    parser.add_argument('--num-train-problems', type=int, default=None,
                       help='Number of problems to use for training (rest used for testing). Default: None = use all for training')
    parser.add_argument('--turn', type=str, default=None,
                       help=f'Turn name (e.g., turn1, turn2, ..., turn10). Default: {DEFAULT_TURN}')
    args = parser.parse_args()

    # Set turn
    turn = args.turn if args.turn else DEFAULT_TURN

    # Set up paths
    parent_dir = f"/home/ubuntu/cactts/big_results/qwen8b/MathArena_aime_2025/{turn}"
    choices_dir = Path(f"{parent_dir}/choices")
    metrics_file = Path(f"{parent_dir}/all_response_metrics.jsonl")

    # Create results directory for this turn
    results_dir = Path(f"/home/ubuntu/cactts/proposed_method/results/{turn}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Logprob Weight Optimization for AIME 2025 - {turn.upper()}")
    print("=" * 80)

    if args.num_problems:
        print(f"Running on SUBSET: first {args.num_problems} problems")
    else:
        print("Running on FULL dataset")

    print(f"Parallelism: {args.num_workers} file workers × {args.num_choice_workers} choice workers")
    if args.subsample_size:
        print(f"Subsampling: {args.subsample_size} rollouts per problem")
    if args.num_train_problems:
        print(f"Train/Test split: {args.num_train_problems} training problems, rest for testing")
    print("=" * 80)
    print("\nStep 1: Loading data...")
    print("=" * 80)
    sys.stdout.flush()

    all_data, metrics = load_data_parallel(choices_dir=choices_dir,
                                           metrics_file=metrics_file,
                                           turn=turn,
                                           num_problems=args.num_problems,
                                           num_workers=args.num_workers,
                                           num_choice_workers=args.num_choice_workers,
                                           subsample_size=args.subsample_size)


    # Determine number of top_logprobs
    num_top_logprobs = all_data[0]['logprobs'].shape[1] if all_data else 10
    print(f"\nNumber of top logprobs per token: {num_top_logprobs}")
    sys.stdout.flush()

    # Split data into train and test if requested
    train_data = all_data
    test_data = []

    if args.num_train_problems is not None:
        import random
        random.seed(42)  # For reproducibility

        print("\n" + "=" * 80)
        print("Step 2: Train/Test Split")
        print("=" * 80)

        # Group data by problem_id
        problems_dict = defaultdict(list)
        for item in all_data:
            problems_dict[item['problem_id']].append(item)

        problem_ids = sorted(problems_dict.keys())

        if args.num_train_problems >= len(problem_ids):
            print(f"Warning: num_train_problems ({args.num_train_problems}) >= total problems ({len(problem_ids)})")
            print("Using all problems for training, no test set.")
            train_data = all_data
            test_data = []
        else:
            # Randomly select training problems
            train_problem_ids = random.sample(problem_ids, args.num_train_problems)
            test_problem_ids = [pid for pid in problem_ids if pid not in train_problem_ids]

            # Split data
            train_data = []
            test_data = []
            for item in all_data:
                if item['problem_id'] in train_problem_ids:
                    train_data.append(item)
                else:
                    test_data.append(item)

            print(f"Training set: {len(train_problem_ids)} problems, {len(train_data)} rollouts")
            print(f"Test set: {len(test_problem_ids)} problems, {len(test_data)} rollouts")
            sys.stdout.flush()

    print("\n" + "=" * 80)
    print(f"Step 3: Baseline Performance (Training Set)")
    print("=" * 80)

    # Random selection baseline on training data
    random_baseline = compute_random_baseline(train_data, num_trials=100)

    # Uniform weights baseline on training data
    baseline = compute_baseline_performance(train_data)
    print(f"\n✓ Uniform Weights Baseline: {baseline['accuracy']:.2%} ({baseline['correct_count']}/{baseline['total_count']})")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("Step 4: Setting up optimization...")
    print("=" * 80)

    problem, weights_var, problems = setup_optimization(train_data, num_top_logprobs)

    print("\n" + "=" * 80)
    print("Step 5: Solving optimization problem...")
    print("=" * 80)
    print(f"Solver: {args.solver}")
    sys.stdout.flush()

    # Solve the problem with verbose output
    solve_start = time()
    problem.solve(verbose=True, solver=getattr(cp, args.solver), max_iters=10000)
    solve_time = time() - solve_start

    print(f"\nSolving took: {solve_time:.2f}s")
    print(f"Optimization status: {problem.status}")
    print(f"Optimal value: {problem.value:.6f}")
    sys.stdout.flush()

    if weights_var.value is not None:
        learned_weights = weights_var.value

        print("\n" + "=" * 80)
        print("Step 6: Results")
        print("=" * 80)

        print(f"\nLearned weights (sum={np.sum(learned_weights):.6f}):")
        for i, w in enumerate(learned_weights):
            print(f"  Top-{i+1} logprob: {w:.6f}")
        sys.stdout.flush()

        # Evaluate on training set
        optimized_train = evaluate_weights(train_data, learned_weights)

        print(f"\n{'='*80}")
        print("TRAINING SET PERFORMANCE")
        print(f"{'='*80}")
        print(f"\n{'Metric':<20} {'Random':<20} {'Baseline':<20} {'Optimized':<20} {'Gain':<15}")
        print("-" * 95)
        print(f"{'Accuracy':<20} {random_baseline['mean_accuracy']:.2%}±{random_baseline['std_accuracy']:.2%}          {baseline['accuracy']:.2%} ({baseline['correct_count']}/{baseline['total_count']})       {optimized_train['accuracy']:.2%} ({optimized_train['correct_count']}/{optimized_train['total_count']})       {(optimized_train['accuracy'] - baseline['accuracy']) * 100:+.2f}pp")

        # Evaluate on test set if available
        test_baseline = None
        optimized_test = None
        test_random_baseline = None
        if test_data:
            print(f"\n{'='*80}")
            print("TEST SET PERFORMANCE")
            print(f"{'='*80}")

            # Compute test baselines
            test_random_baseline = compute_random_baseline(test_data, num_trials=100)
            test_baseline = compute_baseline_performance(test_data)
            optimized_test = evaluate_weights(test_data, learned_weights)

            print(f"\n{'Metric':<20} {'Random':<20} {'Baseline':<20} {'Optimized':<20} {'Gain':<15}")
            print("-" * 95)
            print(f"{'Accuracy':<20} {test_random_baseline['mean_accuracy']:.2%}±{test_random_baseline['std_accuracy']:.2%}          {test_baseline['accuracy']:.2%} ({test_baseline['correct_count']}/{test_baseline['total_count']})       {optimized_test['accuracy']:.2%} ({optimized_test['correct_count']}/{optimized_test['total_count']})       {(optimized_test['accuracy'] - test_baseline['accuracy']) * 100:+.2f}pp")

        # Save results
        suffix = f"_{args.num_problems}prob" if args.num_problems else "_full"
        if args.subsample_size:
            suffix += f"_sub{args.subsample_size}"
        if args.num_train_problems:
            suffix += f"_train{args.num_train_problems}"

        weights_file = results_dir / f"learned_logprob_weights{suffix}.npy"
        np.save(weights_file, learned_weights)
        print(f"\n✓ Weights saved to: {weights_file}")

        results_file = results_dir / f"optimization_results{suffix}.json"
        results = {
            'config': {
                'num_problems': args.num_problems,
                'num_train_problems': args.num_train_problems,
                'subsample_size': args.subsample_size,
                'solver': args.solver
            },
            'train': {
                'random_baseline': {
                    'mean_accuracy': random_baseline['mean_accuracy'],
                    'std_accuracy': random_baseline['std_accuracy'],
                    'num_trials': random_baseline['num_trials']
                },
                'baseline': {
                    'accuracy': baseline['accuracy'],
                    'correct_count': baseline['correct_count'],
                    'total_count': baseline['total_count'],
                    'total_correct_rollouts': baseline['total_correct_rollouts'],
                    'total_rollouts': baseline['total_rollouts']
                },
                'optimized': {
                    'accuracy': optimized_train['accuracy'],
                    'correct_count': optimized_train['correct_count'],
                    'total_count': optimized_train['total_count'],
                    'problem_results': optimized_train['problem_results']
                }
            },
            'learned_weights': learned_weights.tolist(),
            'solve_time': solve_time
        }

        # Add test results if available
        if test_data:
            results['test'] = {
                'random_baseline': {
                    'mean_accuracy': test_random_baseline['mean_accuracy'],
                    'std_accuracy': test_random_baseline['std_accuracy'],
                    'num_trials': test_random_baseline['num_trials']
                },
                'baseline': {
                    'accuracy': test_baseline['accuracy'],
                    'correct_count': test_baseline['correct_count'],
                    'total_count': test_baseline['total_count'],
                    'total_correct_rollouts': test_baseline['total_correct_rollouts'],
                    'total_rollouts': test_baseline['total_rollouts']
                },
                'optimized': {
                    'accuracy': optimized_test['accuracy'],
                    'correct_count': optimized_test['correct_count'],
                    'total_count': optimized_test['total_count'],
                    'problem_results': optimized_test['problem_results']
                }
            }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {results_file}")

        print("\n" + "=" * 80)
        print("DONE!")
        print("=" * 80)

    else:
        print("\n❌ ERROR: Optimization failed to find a solution")
        print(f"Problem status: {problem.status}")
        sys.exit(1)

if __name__ == "__main__":
    main()
