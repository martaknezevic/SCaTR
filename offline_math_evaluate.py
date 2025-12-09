"""
Offline Math Evaluation Script

Evaluates math problems using pre-generated responses stored on disk.
Uses the modular MathEvaluator framework with configuration files.

Usage:
    python offline_math_evaluate.py --config config_aime.yaml --choices_dir ./results/choices
    python offline_math_evaluate.py --config config_math.yaml --choices_dir ./math_choices --output_dir ./eval_results
"""

import time
import asyncio
import argparse
import os
import random
import numpy as np
import yaml
from math_evaluator import MathEvaluator, EvaluationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Math Evaluation")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--choices_dir", type=str, required=True,
                       help="Directory containing pre-generated choices")
    
    # Optional overrides
    parser.add_argument("--output_dir", type=str,
                       help="Output directory (overrides config)")
    parser.add_argument("--tail_n", type=int,
                       help="Tail confidence window size (overrides config)")
    parser.add_argument("--group_size", type=int,
                       help="Group size for metrics (overrides config)")
    parser.add_argument("--max_concurrent", type=int,
                       help="Max concurrent problems (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


async def main():
    print("="*80)
    print("OFFLINE MATH EVALUATION")
    print("="*80)
    
    args = parse_args()
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = EvaluationConfig.from_yaml(args.config)
    
    # Apply command-line overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.tail_n:
        config.tail_n = args.tail_n
    if args.group_size:
        config.group_size = args.group_size
    if args.max_concurrent:
        config.max_concurrent = args.max_concurrent
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize evaluator (no OpenAI client needed for offline evaluation)
    client = None
    evaluator = MathEvaluator(config, client)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.dataset_source}")
    print(f"  Split: {config.dataset_split}")
    print(f"  Loader: {evaluator.loader.get_name()}")
    print(f"  Extractor: {evaluator.extractor.get_name()}")
    print(f"  Grader: {evaluator.grader.get_name()}")
    print(f"  Choices directory: {args.choices_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Tail-N: {config.tail_n}")
    print(f"  Group size: {config.group_size}")
    print(f"  Max concurrent: {config.max_concurrent}")
    
    # Verify choices directory exists
    if not os.path.exists(args.choices_dir):
        print(f"\n❌ ERROR: Choices directory does not exist: {args.choices_dir}")
        return
    
    # Count choice files
    choice_files = [f for f in os.listdir(args.choices_dir) if f.endswith('_choices.pkl')]
    print(f"\nFound {len(choice_files)} choice files in {args.choices_dir}")
    
    # Run offline evaluation
    print("\n" + "="*80)
    print("STARTING OFFLINE EVALUATION")
    print("="*80)
    
    start_time = time.time()
    
    try:
        results = await evaluator.evaluate_dataset_offline(args.choices_dir)
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total time: {execution_time:.1f}s ({execution_time/60:.1f} min)")
    print(f"Problems evaluated: {results['dataset_info']['total_problems']}")
    
    strategy_results = results['strategy_results']['strategy_results']
    if strategy_results:
        # Find best strategy
        best = max(strategy_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best strategy: {best[0]}")
        print(f"   Accuracy: {best[1]['accuracy']:.2%}")
        print(f"   Correct: {best[1]['correct']}/{best[1]['total']}")
        
        # Show top 5 strategies
        print(f"\n📊 Top 5 strategies:")
        sorted_strategies = sorted(strategy_results.items(), 
                                  key=lambda x: x[1]['accuracy'], 
                                  reverse=True)
        for i, (name, stats) in enumerate(sorted_strategies[:5], 1):
            print(f"   {i}. {name}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    print(f"\n📁 Results saved to: {config.output_dir}")
    print(f"   - Strategy results: {evaluator.strategies_file}")
    print(f"   - Response metrics: {evaluator.metrics_file}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
