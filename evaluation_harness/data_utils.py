import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from time import time
import sys


def load_metrics(metrics_file: Path) -> Dict[Tuple[str, int], Dict]:
    """Load metrics JSONL file."""
    print(f"Loading metrics from {metrics_file}...")
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

def _extract_message(choice):
        """Extract full text message from a vLLM/OpenAI-style choice object."""
        if not hasattr(choice, "logprobs") or not hasattr(choice.logprobs, "content"):
            return ""
        return "".join(cct.token for cct in choice.logprobs.content if hasattr(cct, "token"))

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

    # Extract logprobs
    logprobs = extract_top_logprobs_from_choice(choice)
    
    # Get max logprobs for each token
    max_logprobs = np.max(logprobs, axis=1) if logprobs.size > 0 else np.array([])
    # compute average logprobs based on rolling window
    window_size = 256
    if max_logprobs.size > window_size:
        avg_logprobs = np.convolve(max_logprobs, np.ones(window_size)/window_size, mode='valid')
    else:
        avg_logprobs = max_logprobs

    if logprobs.size == 0:
        return rollout_idx, None
    
    message = _extract_message(choice)

    result = {
        'problem_id': problem_id,
        'rollout_idx': rollout_idx,
        'logprobs': logprobs,
        'avg_logprobs': avg_logprobs,
        'answer': message,
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

    # Process choices in parallel
    results_dict = {}
    with ThreadPoolExecutor(max_workers=num_choice_workers) as executor:
        futures = {executor.submit(process_single_choice, arg): arg[0] for arg in choice_args}

        for future in as_completed(futures):
            rollout_idx, result = future.result()
            if result is not None:
                results_dict[rollout_idx] = result

    # Sort results by rollout_idx
    results = [results_dict[idx] for idx in sorted(results_dict.keys())]

    process_time = time() - start_time
    return problem_id, results, process_time


def load_data_parallel(choices_dir: Path,
                       metrics_file: Path,
                       dataset_name: str,
                       num_problems: int = None,
                       num_workers: int = 32,
                       num_choice_workers: int = 8,
                       use_cache: bool = True,
                       subsample_size: int = None,
                       cache_dir: Path = None) -> Tuple[List[Dict], Dict]:
    """
    Load pickle files in parallel with caching.

    Args:
        choices_dir: Directory containing pickle files
        metrics_file: Path to metrics JSONL file
        dataset_name: Name of dataset (for cache file naming)
        num_problems: If specified, only load first N problems
        num_workers: Number of parallel workers for pickle files
        num_choice_workers: Number of parallel workers per pickle file
        use_cache: If True, save/load processed data from cache
        subsample_size: If specified, randomly subsample this many rollouts per problem
        cache_dir: Directory to store cache files

    Returns:
        Tuple of (all_data, metrics)
    """
    # Define cache file path with hierarchical directory structure
    # Structure: cache/gptoss/MathArena_aime_2025/turn1_processed_data_full.pkl
    if cache_dir is None:
        cache_dir = Path("/home/ubuntu/cactts/evaluation_harness/cache")

    # Extract dataset identifiers from path
    # Example: /path/to/gptoss/MathArena_aime_2025/turn1 -> gptoss, MathArena_aime_2025
    try:
        parent_path = Path(choices_dir).parent
        path_parts = parent_path.parts

        if 'GY2233_AIME-2024-2025' in path_parts:
            idx = path_parts.index('GY2233_AIME-2024-2025')
            # Get the identifier before MathArena (e.g., "gptoss")
            dataset_base = path_parts[idx - 2] if idx > 0 else "default"
            # Use MathArena_aime_2025 as the dataset name
            dataset_base = f'{path_parts[-4]}_{path_parts[-1]}'
            dataset_collection = path_parts[-2]
        else:
            # Fallback: use last two parts of parent path
            dataset_base = path_parts[-2] if len(path_parts) >= 2 else "default"
            dataset_collection = path_parts[-1] if len(path_parts) >= 1 else "data"
    except:
        dataset_base = "default"
        dataset_collection = "data"

    # Create hierarchical cache directory: cache/gptoss/MathArena_aime_2025/
    dataset_cache_dir = cache_dir / dataset_base / dataset_collection
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache filename with turn name
    cache_suffix = f"_{num_problems}prob" if num_problems else "_full"
    if subsample_size:
        cache_suffix += f"_sub{subsample_size}"
    cache_file = dataset_cache_dir / f"{dataset_name}_processed_data{cache_suffix}.pkl"

    # Also define the full cache file (without subsampling/num_problems)
    full_cache_file = dataset_cache_dir / f"{dataset_name}_processed_data_full.pkl"

    # Try to load from exact cache match
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

    # If exact cache not found but we only need subsampling and full cache exists, use it
    if (use_cache and subsample_size is not None and not num_problems and
        full_cache_file.exists() and not cache_file.exists()):
        print(f"\n✓ Found full cached data at {full_cache_file}")
        print("Loading full cache for subsampling...")
        sys.stdout.flush()
        start = time()
        with open(full_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        load_time = time() - start
        print(f"✓ Loaded {len(cache_data['all_data'])} rollouts from cache in {load_time:.2f}s")
        sys.stdout.flush()

        all_data = cache_data['all_data']
        metrics = cache_data['metrics']

        # Perform subsampling
        import random
        random.seed(42)

        print(f"\nSubsampling {subsample_size} rollouts per problem...")
        sys.stdout.flush()

        problems = defaultdict(list)
        for item in all_data:
            problems[item['problem_id']].append(item)

        subsampled_data = []
        for problem_id, rollouts in problems.items():
            if len(rollouts) <= subsample_size:
                subsampled_data.extend(rollouts)
            else:
                subsampled_data.extend(random.sample(rollouts, subsample_size))

        print(f"After subsampling: {len(subsampled_data)} rollouts (from {len(all_data)})")
        sys.stdout.flush()

        # Save subsampled cache
        print(f"\nSaving subsampled data to cache: {cache_file}")
        sys.stdout.flush()
        with open(cache_file, 'wb') as f:
            pickle.dump({'all_data': subsampled_data, 'metrics': metrics}, f)
        print("✓ Cache saved")
        sys.stdout.flush()

        return subsampled_data, metrics

    # If no cache, process data
    print("\nNo cache found, processing data from scratch...")
    sys.stdout.flush()

    start = time()
    metrics = load_metrics(metrics_file)

    # Get all pickle files
    pickle_files = sorted(choices_dir.glob("*_choices.pkl"))

    if num_problems:
        pickle_files = pickle_files[:num_problems]
        print(f"\nProcessing subset: first {num_problems} problems")

    print(f"Found {len(pickle_files)} pickle files to process")
    print(f"Using {num_workers} parallel file workers, {num_choice_workers} choice workers per file")
    sys.stdout.flush()

    all_data = []

    # Prepare arguments for parallel processing
    args_list = [(pkl_file, metrics, num_choice_workers) for pkl_file in pickle_files] 
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_single_pickle, args): args[0]
                          for args in args_list}

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
        random.seed(42)

        print(f"\nSubsampling {subsample_size} rollouts per problem...")
        sys.stdout.flush()

        problems = defaultdict(list)
        for item in all_data:
            problems[item['problem_id']].append(item)

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


def split_data(all_data: List[Dict],
               num_train_problems: int = None,
               num_val_problems: int = None,
               num_test_problems: int = None,
               random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train, validation, and test sets by problem_id.

    Args:
        all_data: List of all data items
        num_train_problems: Number of problems for training (None = 10% of total)
        num_val_problems: Number of problems for validation (None = 10% of total)
        num_test_problems: Number of problems for test (None = 10% of total)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    import random
    random.seed(random_seed)

    # Group by problem_id
    problems_dict = defaultdict(list)
    for item in all_data:
        problems_dict[item['problem_id']].append(item)

    problem_ids = sorted(problems_dict.keys())
    total_problems = len(problem_ids)

    # Determine counts
    if num_val_problems is None:
        val_count = max(1, int(total_problems * 0.1))
    else:
        val_count = min(num_val_problems, total_problems)

    if num_test_problems is None:
        test_count = max(1, int(total_problems * 0.1))
    else:
        test_count = min(num_test_problems, total_problems)

    if num_train_problems is None:
        train_count = int(total_problems * 0.1)
        train_count = min(train_count, total_problems - val_count - test_count)
    else:
        train_count = min(num_train_problems, total_problems - val_count - test_count)

    # Shuffle and split problem IDs
    shuffled_ids = problem_ids.copy()
    random.shuffle(shuffled_ids)

    train_problem_ids = set(shuffled_ids[:train_count])
    val_problem_ids = set(shuffled_ids[train_count:train_count + val_count])
    test_problem_ids = set(shuffled_ids[train_count + val_count:train_count + val_count + test_count])

    # Split data
    train_data = []
    val_data = []
    test_data = []

    for item in all_data:
        if item['problem_id'] in train_problem_ids:
            train_data.append(item)
        elif item['problem_id'] in val_problem_ids:
            val_data.append(item)
        elif item['problem_id'] in test_problem_ids:
            test_data.append(item)


    print(f"\nData split (seed={random_seed}):")
    print(f"  Training: {len(train_problem_ids)} problems, {len(train_data)} rollouts")
    # print num of correct rollouts per problem id in train set
    correct_counts = defaultdict(int)
    for item in train_data:
        if item['is_correct']:
            correct_counts[item['problem_id']] += 1
    # for pid in train_problem_ids:
    #     print(f"    Problem {pid}: {correct_counts[pid]} correct rollouts")
    print(f"  Validation: {len(val_problem_ids)} problems, {len(val_data)} rollouts")
    print(f"  Test: {len(test_problem_ids)} problems, {len(test_data)} rollouts")
    sys.stdout.flush()

    return train_data, val_data, test_data