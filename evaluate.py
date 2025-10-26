import time
import asyncio
import pickle
import argparse
import os
from cactts_aime25 import AIMEEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="AIME Evaluator Configuration")
    
    # Dataset
    parser.add_argument("--dataset_source", type=str, default="MathArena/aime_2025")
    parser.add_argument("--split", type=str, default="train")
    
    # Experimental settings - make these optional later
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--turn", type=int)
    
    
    # Evaluation
    parser.add_argument("--choices_dir", type=str, default="/home/ubuntu/cactts/big_results_partial/qwen8b/MathArena_aime_2025/turn1/choices")
    parser.add_argument("--tail_n", type=int, default=2048)
    parser.add_argument("--topk_logprobs", type=int, default=10)
    parser.add_argument("--group_size", type=int, default=1024)
      
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/test")
    
    return parser.parse_args()


async def main():
    print("AIME 2025 Evaluation Script")
    print("="*50)
    
    args = parse_args()
    
    # Initialize evaluator
    client = None
    model_id = None
    metrics_file = f'confidence_metrics_{args.model_name}_{args.turn}_top{args.topk_logprobs}logprobs_tail{args.tail_n}_group{args.group_size}.jsonl'
    strategies_file = f'strategies_{args.model_name}_{args.turn}_top{args.topk_logprobs}logprobs_tail{args.tail_n}_group{args.group_size}.jsonl'
    evaluator = AIMEEvaluator(client, model_id, args.output_dir, metrics_file=metrics_file, strategies_file=strategies_file)
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING AIME 2025 TASK PERFORMANCE EVALUATION")
    print("Goal: Measure how many problems each strategy solves correctly")
    print("="*80)
    
    start_time = time.time()
    
    results = await evaluator.evaluate_dataset_offline(
        dataset_source=args.dataset_source,
        stored_choices_dir=args.choices_dir,
        topk_logprobs=args.topk_logprobs,
        split=args.split,
        tail_n=args.tail_n,
        group_size=1024
    )
    
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
        best = max(strategy_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest strategy: {best[0]}")
        print(f"  Accuracy: {best[1]['accuracy']:.2%}")
        print(f"  Correct: {best[1]['correct']}/{best[1]['total']}")
    
    print(f"\n📁 Results saved to: {args.output_dir}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())