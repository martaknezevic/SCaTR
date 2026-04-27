#!/usr/bin/env python3
"""
Data utilities for loading and caching processed rollout data.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from time import time
import sys
import hashlib


def get_cache_path(base_dir: Path, dataset_name: str, model_name: str, turn: int) -> Path:
    """
    Generate cache file path based on dataset, model, and turn.
    
    Args:
        base_dir: Base cache directory
        dataset_name: Name of dataset (e.g., 'humaneval', 'kodcode')
        model_name: Name of model (e.g., 'gptoss', 'olmo7b')
        turn: Turn number
    
    Returns:
        Path to cache file
    """
    cache_dir = base_dir / model_name / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"turn{turn}_processed.pkl"
    return cache_file


def load_metrics(metrics_file: Path) -> Dict[Tuple[str, int], Dict]:
    """Load metrics JSONL file."""
    print(f"Loading metrics from {metrics_file}...")
    sys.stdout.flush()
    
    metrics = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            prob_id = data['problem_id']
            if '/' in prob_id:
                data['problem_id'] = prob_id.replace('/', '_')
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


def process_single_choice(args: Tuple) -> Tuple[int, Optional[Dict]]:
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
    
    if logprobs.size == 0:
        return rollout_idx, None
    
    message = _extract_message(choice)

    result = {
        'problem_id': problem_id,
        'rollout_idx': rollout_idx,
        'logprobs': logprobs,
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


def load_dataset(data_dir: Path, dataset_name: str, model_name: str, turn: int,
                num_workers: int = 32, num_choice_workers: int = 4,
                cache_dir: Optional[Path] = None) -> List[Dict]:
    """
    Load dataset with automatic caching.
    
    Args:
        data_dir: Base data directory (e.g., /tmp/scatr/data)
        dataset_name: Dataset name (e.g., 'humaneval', 'mbpp')
        model_name: Model name (e.g., 'gptoss', 'olmo7b')
        turn: Turn number
        num_workers: Number of parallel workers for pickle files
        num_choice_workers: Number of parallel workers per pickle file
        cache_dir: Cache directory (default: ./cache)
    
    Returns:
        List of processed data items
    """
    if cache_dir is None:
        cache_dir = Path("./cache")
    
    cache_file = get_cache_path(cache_dir, dataset_name, model_name, turn)
    
    # Try to load from cache
    if cache_file.exists():
        print(f"\n✓ Found cached data at {cache_file}")
        print("Loading from cache...")
        sys.stdout.flush()
        start = time()
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        load_time = time() - start
        print(f"✓ Loaded {len(data)} rollouts from cache in {load_time:.2f}s")
        sys.stdout.flush()
        return data
    
    # Load from scratch
    print(f"\nNo cache found, processing data from scratch...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model: {model_name}")
    print(f"  Turn: {turn}")
    sys.stdout.flush()
    
    # Construct paths
    dataset_dir = data_dir / model_name / dataset_name / f"turn{turn}"
    choices_dir = dataset_dir / "choices"
    metrics_file = dataset_dir / "all_response_metrics.jsonl"
    
    if not choices_dir.exists():
        raise FileNotFoundError(f"Choices directory not found: {choices_dir}")
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    start = time()
    metrics = load_metrics(metrics_file)

    # Get all pickle files
    pickle_files = sorted(choices_dir.glob("*_choices.pkl"))
    print(f"\nProcessing {len(pickle_files)} pickle files from {choices_dir}...")
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
                if completed % 10 == 0 or completed == len(pickle_files):
                    print(f"[{completed}/{len(pickle_files)}] ✓ {problem_id}: {len(results)} rollouts ({process_time:.2f}s)")
                    sys.stdout.flush()
            except Exception as e:
                print(f"[{completed}/{len(pickle_files)}] ✗ {pkl_file.stem}: Error - {e}")
                sys.stdout.flush()

    total_time = time() - start
    print(f"\nTotal loading time: {total_time:.2f}s")
    print(f"Loaded {len(all_data)} valid rollouts")
    sys.stdout.flush()

    # Save to cache
    print(f"\nSaving processed data to cache: {cache_file}")
    sys.stdout.flush()
    with open(cache_file, 'wb') as f:
        pickle.dump(all_data, f)
    print("✓ Cache saved")
    sys.stdout.flush()
    
    return all_data


def split_train_val(train_data: List[Dict], val_ratio: float = 0.2, 
                    random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split training data into train and validation sets by problem_id.
    
    Args:
        train_data: List of training data items
        val_ratio: Ratio of problems to use for validation
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data)
    """
    import random
    random.seed(random_seed)
    
    # Group by problem_id
    problems_dict = defaultdict(list)
    for item in train_data:
        problems_dict[item['problem_id']].append(item)
    
    problem_ids = sorted(problems_dict.keys())
    total_problems = len(problem_ids)
    
    # Determine val count
    val_count = max(1, int(total_problems * val_ratio))
    
    # Shuffle and split problem IDs
    shuffled_ids = problem_ids.copy()
    random.shuffle(shuffled_ids)
    
    train_problem_ids = set(shuffled_ids[val_count:])
    val_problem_ids = set(shuffled_ids[:val_count])
    
    # Split data
    train_split = []
    val_split = []
    
    for item in train_data:
        if item['problem_id'] in train_problem_ids:
            train_split.append(item)
        elif item['problem_id'] in val_problem_ids:
            val_split.append(item)
    
    print(f"\nTrain/Val split (seed={random_seed}):")
    print(f"  Training: {len(train_problem_ids)} problems, {len(train_split)} rollouts")
    print(f"  Validation: {len(val_problem_ids)} problems, {len(val_split)} rollouts")
    sys.stdout.flush()
    
    return train_split, val_split


def get_label_distribution(data: List[Dict]) -> Dict[str, float]:
    """Get distribution of correct/incorrect labels."""
    total = len(data)
    correct = sum(1 for item in data if item['is_correct'])
    incorrect = total - correct
    
    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'correct_ratio': correct / total if total > 0 else 0.0,
        'incorrect_ratio': incorrect / total if total > 0 else 0.0
    }


def compute_class_weights(data: List[Dict]) -> Tuple[float, float]:
    """
    Compute class weights for balanced loss.
    Returns (weight_for_incorrect, weight_for_correct).
    """
    dist = get_label_distribution(data)
    
    if dist['correct'] == 0 or dist['incorrect'] == 0:
        return 1.0, 1.0
    
    # Weight inversely proportional to class frequency
    total = dist['total']
    weight_incorrect = total / (2 * dist['incorrect'])
    weight_correct = total / (2 * dist['correct'])
    
    return weight_incorrect, weight_correct