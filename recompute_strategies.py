#!/usr/bin/env python3
"""
Recompute selection strategies from a JSONL metrics file.

Usage:
    python recompute_strategies.py <path_to_metrics.jsonl> [--output strats.json]
"""

import argparse
import json
import random
from collections import defaultdict
from typing import Dict, List, Any
from pathlib import Path


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_problem(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by problem_id."""
    grouped = defaultdict(list)
    for record in records:
        problem_id = record['problem_id']
        grouped[problem_id].append(record)
    return dict(grouped)


def get_metric_keys() -> List[str]:
    """Generate all metric key combinations."""
    metrics = ['mean', 'median', 'variance', 'gap', 'entropy', 'gap_probs']
    types = ['full', 'tail', 'group_lowest', 'group_highest', 'group_bottom_pct', 'group_top_pct']
    
    metric_keys = []
    for metric in metrics:
        for type_ in types:
            metric_keys.append(f"{metric}_{type_}")
    
    return metric_keys


def apply_selection_strategies(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Apply different selection strategies to a list of results for a single problem.
    
    Args:
        results: List of result dictionaries with fields like is_correct, mean_tail, etc.
    
    Returns:
        Dictionary mapping strategy name to the selected result
    """
    strategies = {}
    
    if not results:
        return strategies
    
    # Random selection
    strategies['random'] = random.choice(results)
    
    # Oracle selection (best by reward if any passing, else best by some metric)
    passing_results = [r for r in results if r.get('is_correct', False)]
    if passing_results:
        # Pick the passing result with highest mean_tail (or first passing one)
        strategies['oracle'] = max(
            passing_results, 
            key=lambda x: x.get('mean_tail', 0)
        )
    else:
        # No passing results, pick best by mean_tail metric
        strategies['oracle'] = max(
            results, 
            key=lambda x: x.get('mean_tail', 0)
        )
    
    # Get all metric keys
    metric_keys = get_metric_keys()
    
    # Filter to only keys that exist in the results
    if results:
        available_keys = [key for key in metric_keys if key in results[0]]
        
        for metric_key in available_keys:
            # Highest metric value
            strategies[f'highest_{metric_key}'] = max(
                results,
                key=lambda x: x.get(metric_key, 0)
            )
            
            # Lowest metric value
            strategies[f'lowest_{metric_key}'] = min(
                results,
                key=lambda x: x.get(metric_key, 0)
            )
    
    return strategies


def compute_strategy_accuracies(
    grouped_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute accuracy for each selection strategy across all problems.
    
    Args:
        grouped_results: Dictionary mapping problem_id to list of results
    
    Returns:
        Dictionary mapping strategy name to {correct, total, accuracy}
    """
    strategy_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for problem_id, results in grouped_results.items():
        if not results:
            continue
        
        # Apply selection strategies for this problem
        strategies = apply_selection_strategies(results)
        
        # Update statistics for each strategy
        for strategy_name, selected_result in strategies.items():
            strategy_stats[strategy_name]['total'] += 1
            if selected_result.get('is_correct', False):
                strategy_stats[strategy_name]['correct'] += 1
    
    # Compute accuracies
    strategy_results = {}
    for strategy_name, stats in strategy_stats.items():
        total = stats['total']
        correct = stats['correct']
        accuracy = correct / total if total > 0 else 0.0
        
        strategy_results[strategy_name] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
    
    return strategy_results


def extract_dataset_info(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract dataset information from records.
    
    This tries to infer metadata from the data itself.
    """
    if not records:
        return {}
    
    # Count unique problems
    unique_problems = len(set(r['problem_id'] for r in records))
    
    # Count rollouts per problem (use first problem as sample)
    first_problem_id = records[0]['problem_id']
    n_gen = sum(1 for r in records if r['problem_id'] == first_problem_id)
    
    # Try to extract other info if available
    dataset_info = {
        'source': 'unknown',
        'split': 'test',
        'loader': 'unknown',
        'extractor': 'unknown',
        'grader': 'unknown',
        'total_problems': unique_problems,
        'n_gen': n_gen,
        'tail_n': 2048,  # Default, may be overridden
        'group_size': 1024,  # Default, may be overridden
        'temperature': 0.6  # Default
    }
    
    return dataset_info


def main():
    parser = argparse.ArgumentParser(
        description='Recompute selection strategies from JSONL metrics file'
    )
    parser.add_argument(
        'metrics_file',
        type=str,
        help='Path to the JSONL metrics file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='strats.json',
        help='Output JSON file path (default: strats.json)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--dataset-info',
        type=str,
        help='Optional JSON file with dataset info to include in output'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load metrics
    print(f"Loading metrics from {args.metrics_file}...")
    records = load_jsonl(args.metrics_file)
    print(f"Loaded {len(records)} records")
    
    # Group by problem
    print("Grouping by problem_id...")
    grouped = group_by_problem(records)
    print(f"Found {len(grouped)} unique problems")
    
    # Compute strategy accuracies
    print("Computing selection strategy accuracies...")
    strategy_results = compute_strategy_accuracies(grouped)
    
    # Extract or load dataset info
    if args.dataset_info and Path(args.dataset_info).exists():
        with open(args.dataset_info, 'r') as f:
            dataset_info = json.load(f)
    else:
        # Try to load from existing strategy_results.json in the same directory
        metrics_dir = Path(args.metrics_file).parent
        existing_strats = metrics_dir / 'strategy_results.json'
        
        if existing_strats.exists():
            print(f"Loading dataset_info from {existing_strats}...")
            with open(existing_strats, 'r') as f:
                existing_data = json.load(f)
                dataset_info = existing_data.get('dataset_info', extract_dataset_info(records))
        else:
            dataset_info = extract_dataset_info(records)
    
    # Prepare output
    output = {
        'dataset_info': dataset_info,
        'strategy_results': strategy_results
    }
    
    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("STRATEGY RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {len(grouped)}")
    print(f"\nTop 5 strategies by accuracy:")
    
    sorted_strategies = sorted(
        strategy_results.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    for i, (name, stats) in enumerate(sorted_strategies[:5], 1):
        print(f"{i}. {name:30s} {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
    
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
