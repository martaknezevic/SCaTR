#!/usr/bin/env python3
"""
Script to run all hyperparameter searches across different methods and evaluation types.

Usage:
    # Run all searches sequentially
    python run_all_hyper_searches.py
    
    # Run specific searches
    python run_all_hyper_searches.py --methods transformer dct_nn --eval-types code
    
    # Dry run to see commands
    python run_all_hyper_searches.py --dry-run
    
    # Run in parallel (use with caution - requires multiple GPUs)
    python run_all_hyper_searches.py --parallel --gpus 0 1 2 3
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import time


# Define all hyperparameter search configurations
HYPER_SEARCH_CONFIGS = [
    # Code evaluation
    {
        "name": "code_transformer",
        "config": "hyper_search_code_transformer.yaml",
        "method": "transformer",
        "eval_type": "code",
        "default_gpu": 0,
        "description": "Transformer method on code (kodcode→humaneval)"
    },
    {
        "name": "code_dct",
        "config": "hyper_search_code_dct.yaml",
        "method": "dct_nn",
        "eval_type": "code",
        "default_gpu": 1,
        "description": "DCT-NN method on code (kodcode→humaneval)"
    },
    {
        "name": "code_attention",
        "config": "hyper_search_code_attention.yaml",
        "method": "cls_attention_topk",
        "eval_type": "code",
        "default_gpu": 2,
        "description": "CLS Attention method on code (kodcode→humaneval)"
    },
    {
        "name": "code_grouped_transformer",
        "config": "hyper_search_code_grouped_transformer.yaml",
        "method": "grouped_transformer",
        "eval_type": "code",
        "default_gpu": 3,
        "description": "Grouped Transformer method on code (kodcode→humaneval)"
    },
    
    # Math evaluation
    {
        "name": "math_transformer",
        "config": "hyper_search_math_transformer.yaml",
        "method": "transformer",
        "eval_type": "math",
        "default_gpu": 0,
        "description": "Transformer method on math (gsm8k→math500)"
    },
    {
        "name": "math_dct",
        "config": "hyper_search_math_dct.yaml",
        "method": "dct_nn",
        "eval_type": "math",
        "default_gpu": 1,
        "description": "DCT-NN method on math (gsm8k→math500)"
    },
    {
        "name": "math_attention",
        "config": "hyper_search_math_attention.yaml",
        "method": "cls_attention_topk",
        "eval_type": "math",
        "default_gpu": 2,
        "description": "CLS Attention method on math (gsm8k→math500)"
    },
    {
        "name": "math_grouped_transformer",
        "config": "hyper_search_math_grouped_transformer.yaml",
        "method": "grouped_transformer",
        "eval_type": "math",
        "default_gpu": 3,
        "description": "Grouped Transformer method on math (gsm8k→math500)"
    },
]


def build_command(config: dict, gpu: int, dry_run: bool = False) -> List[str]:
    """Build the command to run hyperparameter search."""
    cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "python",
        "hyper_search.py",
        "--search-config",
        config["config"]
    ]
    
    if dry_run:
        cmd.append("--dry-run")
    
    return cmd


def run_command(cmd: List[str], config: dict) -> bool:
    """Execute a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    # Build command string
    if cmd[0].startswith("CUDA_VISIBLE_DEVICES="):
        env_var = cmd[0]
        cmd_str = env_var + " " + " ".join(cmd[1:])
    else:
        cmd_str = " ".join(cmd)
    
    print(f"Command: {cmd_str}\n")
    
    try:
        # Handle CUDA_VISIBLE_DEVICES environment variable
        import os
        if cmd[0].startswith("CUDA_VISIBLE_DEVICES="):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = cmd[0].split('=')[1]
            result = subprocess.run(
                cmd[1:],
                env=env,
                check=True,
                cwd=Path(__file__).parent
            )
        else:
            result = subprocess.run(
                cmd,
                check=True,
                cwd=Path(__file__).parent
            )
        
        print(f"\n✓ {config['name']} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {config['name']} failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)


def run_parallel(configs: List[dict], gpus: List[int]) -> dict:
    """Run multiple hyperparameter searches in parallel."""
    import concurrent.futures
    
    print(f"\n{'='*80}")
    print(f"RUNNING {len(configs)} HYPERPARAMETER SEARCHES IN PARALLEL")
    print(f"Using GPUs: {gpus}")
    print(f"{'='*80}\n")
    
    # Assign GPUs round-robin
    for i, config in enumerate(configs):
        config['assigned_gpu'] = gpus[i % len(gpus)]
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        future_to_config = {
            executor.submit(
                run_command,
                build_command(config, config['assigned_gpu']),
                config
            ): config for config in configs
        }
        
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                success = future.result()
                results[config['name']] = success
            except Exception as e:
                print(f"\n✗ {config['name']} failed with exception: {e}")
                results[config['name']] = False
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run all hyperparameter searches for evaluation harness"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["transformer", "dct_nn", "cls_attention_topk", "grouped_transformer"],
        help="Specific methods to search (default: all)"
    )
    
    parser.add_argument(
        "--eval-types",
        nargs="+",
        choices=["code", "math"],
        help="Evaluation types to run (default: all)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run searches in parallel (requires multiple GPUs)"
    )
    
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        help="GPU devices to use (default: use default_gpu from config)"
    )
    
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running even if some searches fail"
    )
    
    args = parser.parse_args()
    
    # Filter configs based on arguments
    configs_to_run = HYPER_SEARCH_CONFIGS
    
    if args.methods:
        configs_to_run = [c for c in configs_to_run if c['method'] in args.methods]
    
    if args.eval_types:
        configs_to_run = [c for c in configs_to_run if c['eval_type'] in args.eval_types]
    
    if not configs_to_run:
        print("No configurations match the specified filters.")
        return 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH RUNNER")
    print(f"{'='*80}")
    print(f"Total searches to run: {len(configs_to_run)}")
    print(f"Mode: {'PARALLEL' if args.parallel else 'SEQUENTIAL'}")
    if args.dry_run:
        print("DRY RUN MODE - Commands will not be executed")
    print(f"{'='*80}\n")
    
    print("Configurations:")
    for i, config in enumerate(configs_to_run, 1):
        gpu = args.gpus[i % len(args.gpus)] if args.gpus else config['default_gpu']
        print(f"  {i}. {config['name']} (GPU {gpu})")
        print(f"     {config['description']}")
    print()
    
    if not args.dry_run:
        response = input("Proceed? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    # Run searches
    start_time = time.time()
    
    if args.parallel:
        if not args.gpus:
            print("Error: --gpus must be specified for parallel execution")
            return 1
        
        results = run_parallel(configs_to_run, args.gpus)
        
    else:
        # Sequential execution
        results = {}
        for i, config in enumerate(configs_to_run, 1):
            gpu = args.gpus[i % len(args.gpus)] if args.gpus else config['default_gpu']
            
            print(f"\n[{i}/{len(configs_to_run)}] Starting {config['name']}...")
            
            cmd = build_command(config, gpu, dry_run=args.dry_run)
            success = run_command(cmd, config)
            results[config['name']] = success
            
            if not success and not args.continue_on_failure and not args.dry_run:
                print("\nStopping due to failure (use --continue-on-failure to continue)")
                break
    
    # Print summary
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\nResults:")
    
    successful = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
