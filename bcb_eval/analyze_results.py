"""
Analyze BigCodeBench evaluation results

Computes statistics and generates reports from evaluation results.

Usage:
    python analyze_results.py --results bcb_evaluation_results.jsonl
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List
import statistics


def load_results(results_path: str) -> List[dict]:
    """Load evaluation results from JSONL file"""
    results = []
    with open(results_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_results(results: List[dict]) -> Dict:
    """Analyze evaluation results and compute statistics"""
    
    # Overall statistics
    total_responses = len(results)
    total_correct = sum(1 for r in results if r['is_correct'])
    
    # Per-problem statistics
    problem_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'tests_passed': [],
        'total_tests': 0
    })
    
    for result in results:
        problem_id = result['problem_id']
        stats = problem_stats[problem_id]
        
        stats['total'] += 1
        if result['is_correct']:
            stats['correct'] += 1
        stats['tests_passed'].append(result['num_tests_passed'])
        stats['total_tests'] = result['total_tests']
    
    # Compute per-problem accuracy
    problem_accuracies = {}
    for problem_id, stats in problem_stats.items():
        problem_accuracies[problem_id] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'total_responses': stats['total'],
            'correct_responses': stats['correct'],
            'avg_tests_passed': statistics.mean(stats['tests_passed']) if stats['tests_passed'] else 0,
            'total_tests': stats['total_tests']
        }
    
    # Find best and worst problems
    sorted_problems = sorted(
        problem_accuracies.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    return {
        'overall': {
            'total_responses': total_responses,
            'total_correct': total_correct,
            'accuracy': total_correct / total_responses if total_responses > 0 else 0,
            'total_problems': len(problem_stats)
        },
        'per_problem': problem_accuracies,
        'best_problems': sorted_problems[:10],
        'worst_problems': sorted_problems[-10:]
    }


def print_analysis(analysis: Dict):
    """Print analysis results"""
    
    print("\n" + "="*80)
    print("BIGCODEBENCH EVALUATION ANALYSIS")
    print("="*80 + "\n")
    
    # Overall statistics
    overall = analysis['overall']
    print("OVERALL STATISTICS:")
    print(f"  Total Responses: {overall['total_responses']}")
    print(f"  Total Correct: {overall['total_correct']}")
    print(f"  Overall Accuracy: {overall['accuracy']*100:.2f}%")
    print(f"  Unique Problems: {overall['total_problems']}")
    
    # Best problems
    print("\n" + "-"*80)
    print("TOP 10 EASIEST PROBLEMS:")
    print("-"*80)
    for problem_id, stats in analysis['best_problems']:
        print(f"  {problem_id}")
        print(f"    Accuracy: {stats['accuracy']*100:.1f}% ({stats['correct_responses']}/{stats['total_responses']})")
        print(f"    Avg Tests Passed: {stats['avg_tests_passed']:.1f}/{stats['total_tests']}")
    
    # Worst problems
    print("\n" + "-"*80)
    print("TOP 10 HARDEST PROBLEMS:")
    print("-"*80)
    for problem_id, stats in analysis['worst_problems']:
        print(f"  {problem_id}")
        print(f"    Accuracy: {stats['accuracy']*100:.1f}% ({stats['correct_responses']}/{stats['total_responses']})")
        print(f"    Avg Tests Passed: {stats['avg_tests_passed']:.1f}/{stats['total_tests']}")
    
    print("\n" + "="*80 + "\n")


def save_analysis(analysis: Dict, output_path: str):
    """Save analysis to JSON file"""
    
    # Convert to serializable format
    serializable = {
        'overall': analysis['overall'],
        'per_problem': analysis['per_problem'],
        'best_problems': [
            {'problem_id': p[0], **p[1]}
            for p in analysis['best_problems']
        ],
        'worst_problems': [
            {'problem_id': p[0], **p[1]}
            for p in analysis['worst_problems']
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"Analysis saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BigCodeBench evaluation results"
    )
    
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save analysis JSON (optional)"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    print(f"Loaded {len(results)} evaluation results")
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print analysis
    print_analysis(analysis)
    
    # Save analysis if requested
    if args.output:
        save_analysis(analysis, args.output)


if __name__ == "__main__":
    main()
