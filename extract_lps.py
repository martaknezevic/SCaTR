#!/usr/bin/env python3
"""
Minimal script to extract per-token logprobs (of the chosen token) from rollout pickle files.

Usage:
    python extract_logprobs.py --train <dir> --test <dir> --turn <int> [--original]

    --train      Path to training data directory
    --test       Path to test data directory
    --turn       Turn number
    --original   Use original data format (logprobs.content with top_logprobs)
                 If omitted, uses new format (logprobs.top_logprobs dict)
"""

import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from time import time
import sys
import os

# ---------------------------------------------------------------------------
# Logprob extractors
# ---------------------------------------------------------------------------

def extract_chosen_logprobs_original(choice) -> np.ndarray:
    """
    Original format: logprobs.content is a list of token objects, each with
    .logprob (the chosen token's logprob) and .top_logprobs (alternatives).

    Returns:
        np.ndarray of shape (num_tokens,) — one scalar logprob per token.
    """
    if isinstance(choice, dict):
        logprobs = choice.get('logprobs')
        if not logprobs or not logprobs.get('content'):
            return np.array([])
        content = logprobs['content']
    else:
        if not choice.logprobs or not choice.logprobs.content:
            return np.array([])
        content = choice.logprobs.content

    lps = []
    for token_logprob in content:
        if isinstance(token_logprob, dict):
            lps.append(token_logprob['logprob'])
        else:
            lps.append(token_logprob.logprob)

    return np.array(lps, dtype=np.float32)


def extract_chosen_logprobs_new(choice) -> np.ndarray:
    """
    New format: logprobs.top_logprobs is a list of dicts {token: logprob}.
    The chosen token is identified by matching against the token sequence in
    logprobs.tokens (if present), otherwise we take the highest-logprob entry
    as a proxy for the chosen token.

    Returns:
        np.ndarray of shape (num_tokens,) — one scalar logprob per token.
    """

    lps = []
    lps = choice.assistant_logprobs

    return np.array(lps, dtype=np.float32)


# ---------------------------------------------------------------------------
# Message extractor (unchanged from original)
# ---------------------------------------------------------------------------

def _extract_message(choice) -> str:
    if isinstance(choice, dict):
        logprobs = choice.get('logprobs') or {}
        content = logprobs.get('content') or []
        return "".join(c['token'] for c in content if isinstance(c, dict) and 'token' in c)
    if not hasattr(choice, "logprobs") or not hasattr(choice.logprobs, "content"):
        return ""
    return "".join(cct.token for cct in choice.logprobs.content if hasattr(cct, "token"))


# ---------------------------------------------------------------------------
# Metrics loader
# ---------------------------------------------------------------------------

def load_metrics(metrics_file: Path) -> Dict[Tuple[str, int], Dict]:
    print(f"Loading metrics from {metrics_file}...")
    sys.stdout.flush()
    metrics = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            prob_id = data['problem_id'].replace('/', '_')
            data['problem_id'] = prob_id
            key = (prob_id, data['rollout_idx'])
            metrics[key] = data
    print(f"Loaded metrics for {len(metrics)} rollouts")
    sys.stdout.flush()
    return metrics


# ---------------------------------------------------------------------------
# Per-choice worker
# ---------------------------------------------------------------------------

def process_single_choice(args: Tuple):
    rollout_idx, choice, problem_id, metrics, use_original = args
    try:
        problem_id = int(problem_id)  # Convert problem_id to int if possible
        if isinstance(problem_id, int):
            problem_id = 'HumanEval_' + str(problem_id)  # Prefix with 'HumanEval_' to match metrics keys
    except ValueError:
        pass  # Keep as string if conversion fails
    key = (problem_id, rollout_idx)
    if key not in metrics:
        return rollout_idx, None

    extractor = extract_chosen_logprobs_original if use_original else extract_chosen_logprobs_new
    logprobs = extractor(choice)

    if logprobs.size == 0:
        return rollout_idx, None

    return rollout_idx, {
        'problem_id': problem_id,
        'rollout_idx': rollout_idx,
        'logprobs': logprobs,           # shape (num_tokens,)
        'answer': _extract_message(choice),
        'is_correct': metrics[key]['is_correct'],
    }


# ---------------------------------------------------------------------------
# Per-pickle worker
# ---------------------------------------------------------------------------

def process_single_pickle(args: Tuple[Path, Dict, int, bool]):
    pkl_file, metrics, num_choice_workers, use_original = args
    t0 = time()
    problem_id = pkl_file.stem.replace("_choices", "")

    with open(pkl_file, 'rb') as f:
        choices = pickle.load(f)
    choice_args = [
        (idx, (ch if use_original else ch['choice']), problem_id, metrics, use_original)
        for idx, ch in enumerate(choices)
    ]

    results_dict = {}
    with ThreadPoolExecutor(max_workers=num_choice_workers) as ex:
        for fut in as_completed({ex.submit(process_single_choice, a): a[0] for a in choice_args}):
            idx, result = fut.result()
            if result is not None:
                results_dict[idx] = result

    results = [results_dict[k] for k in sorted(results_dict)]
    return problem_id, results, time() - t0


# ---------------------------------------------------------------------------
# Dataset loader with caching
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path, model: str, dataset: str, turn: int, use_original: bool,
                 num_workers: int = 32, num_choice_workers: int = 4,
                 cache_dir: Path = Path("./cache_logprobs"), ) -> List[Dict]:
    """Load and cache chosen-token logprobs for a dataset split directory."""

    fmt_tag = "original" if use_original else "ref"
    cache_file = cache_dir / f"{model}_{dataset}_turn{turn}_{fmt_tag}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        print(f"\n✓ Found cache: {cache_file}")
        t0 = time()
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded {len(data)} rollouts in {time()-t0:.2f}s")
        sys.stdout.flush()
        return data

    print(f"\nNo cache — processing from scratch: {data_dir}")
    sys.stdout.flush()
    
    data_dir = Path(data_dir)

    split_dir = data_dir  / f"turn{turn}"
    if not use_original:
        split_dir = split_dir / "ref_correct"
        if dataset == 'humaneval':
            files = len(os.listdir(split_dir / "choices" ))
            print(files)
            if files > 100:
                choices_dir = split_dir / "choices" 
            else:                
                choices_dir = split_dir / "choices" / "HumanEval"
        else:
            choices_dir = split_dir / "choices"
    else:
        choices_dir = split_dir / "choices"
    metrics_file = split_dir / "all_response_metrics.jsonl" if use_original else split_dir / "ref_response_metrics.jsonl"

    if not choices_dir.exists():
        raise FileNotFoundError(f"Choices directory not found: {choices_dir}")
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    t0 = time()
    metrics = load_metrics(metrics_file)

    pkl_files = sorted(choices_dir.glob("*_choices.pkl"))
    print(f"Processing {len(pkl_files)} pickle files  "
          f"[workers: {num_workers} file / {num_choice_workers} choice]")
    sys.stdout.flush()

    all_data: List[Dict] = []
    args_list = [(p, metrics, num_choice_workers, use_original) for p in pkl_files]

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        fut_map = {ex.submit(process_single_pickle, a): a[0] for a in args_list}
        done = 0
        for fut in as_completed(fut_map):
            try:
                problem_id, results, pt = fut.result()
                all_data.extend(results)
                done += 1
                if done % 10 == 0 or done == len(pkl_files):
                    print(f"  [{done}/{len(pkl_files)}] {problem_id}: "
                          f"{len(results)} rollouts ({pt:.2f}s)")
                    sys.stdout.flush()
            except Exception as e:
                print(f"  ERROR {fut_map[fut].stem}: {e}")
                sys.stdout.flush()

    print(f"\nDone — {len(all_data)} rollouts in {time()-t0:.2f}s")
    print(f"Saving cache → {cache_file}")
    sys.stdout.flush()
    with open(cache_file, 'wb') as f:
        pickle.dump(all_data, f)
    print("✓ Cache saved")
    sys.stdout.flush()
    return all_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract chosen-token logprobs from rollout pickles.")
    p.add_argument("--train", required=True, type=str,
                   help="Path to training data directory")
    p.add_argument("--model", required=True, type=str,
                   help="Model name (e.g. gpt-4-0613) for cache naming")

    p.add_argument("--turn",  required=True, type=int,
                   help="Turn number")
    p.add_argument("--original", action="store_true",
                   help="Use original data format (logprobs.content). "
                        "Omit for new format (logprobs.top_logprobs dict).")
    p.add_argument("--cache-dir", type=Path, default=Path("./cache_logprobs"),
                   help="Directory for cached outputs (default: ./cache_logprobs)")
    p.add_argument("--workers", type=int, default=32,
                   help="Parallel file workers (default: 32)")
    p.add_argument("--choice-workers", type=int, default=4,
                   help="Parallel choice workers per file (default: 4)")
    return p.parse_args()


def main():
    args = parse_args()
    fmt = "original" if args.original else "new"
    print(f"\n{'='*60}")
    print(f"  Logprob extraction — format: {fmt.upper()}  turn: {args.turn}")
    print(f"{'='*60}")

    data_dir = '/efs/cactts/' + ('data' if args.original else 'ref_data')
    data_dir = os.path.join(data_dir, args.model, args.train)
    
    train_data = load_dataset(
        data_dir, args.model, args.train, args.turn, args.original,
        args.workers, args.choice_workers, args.cache_dir,
    )


    print(f"\n{'='*60}")
    print(f"  Train rollouts : {len(train_data)}")
    train_correct = sum(1 for d in train_data if d['is_correct'])
    print(f"  Train correct  : {train_correct} / {len(train_data)} "
          f"({100*train_correct/max(1,len(train_data)):.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()