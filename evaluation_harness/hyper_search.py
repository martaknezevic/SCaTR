#!/usr/bin/env python3
"""
Hyperparameter search script for evaluation harness.

This script reads a hyperparameter search config and launches multiple
evaluation runs with different hyperparameter combinations.

Usage:
    python hyper_search.py --search-config hyper_search_config.yaml
    python hyper_search.py --search-config hyper_search_config.yaml --dry-run
"""

import argparse
import itertools
import subprocess
import sys
import yaml
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import time


def load_search_config(config_path: str) -> Dict[str, Any]:
    """Load hyperparameter search configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_combinations(hyperparameters: Dict[str, List[Any]], 
                         strategy: str = 'grid',
                         max_trials: Optional[int] = None,
                         seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate hyperparameter combinations based on search strategy.
    
    Args:
        hyperparameters: Dictionary mapping parameter names to lists of values
        strategy: 'grid' for grid search, 'random' for random search
        max_trials: Maximum number of trials (None = all combinations for grid)
        seed: Random seed for random search
        
    Returns:
        List of hyperparameter dictionaries
    """
    if strategy == 'grid':
        # Generate all combinations
        keys = list(hyperparameters.keys())
        values = [hyperparameters[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        # Convert to list of dicts
        result = [dict(zip(keys, combo)) for combo in combinations]
        
        # Limit if max_trials specified
        if max_trials is not None and len(result) > max_trials:
            random.seed(seed)
            result = random.sample(result, max_trials)
        
        return result
    
    elif strategy == 'random':
        if max_trials is None:
            raise ValueError("max_trials must be specified for random search")
        
        random.seed(seed)
        result = []
        
        for _ in range(max_trials):
            combo = {k: random.choice(v) for k, v in hyperparameters.items()}
            result.append(combo)
        
        return result
    
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")


def build_command(base_config: str,
                 method_name: str,
                 hyperparams: Dict[str, Any],
                 wandb_config: Dict[str, Any],
                 cuda_device: Optional[int] = None) -> List[str]:
    """
    Build command to launch evaluation with specific hyperparameters.
    
    Args:
        base_config: Base configuration file name (without .yaml)
        method_name: Name of the method being tuned
        hyperparams: Dictionary of hyperparameter values
        wandb_config: WandB configuration
        cuda_device: CUDA device to use
        
    Returns:
        Command as list of strings
    """
    cmd = []
    
    # Set CUDA device
    if cuda_device is not None:
        cmd = [f"CUDA_VISIBLE_DEVICES={cuda_device}"]
    
    # Python command
    cmd.extend(["python", "main.py", f"--config-name={base_config}"])
    
    # Add hyperparameter overrides
    for param, value in hyperparams.items():
        cmd.append(f"methods.0.params.{param}={value}")
    
    # Add WandB config
    if wandb_config.get('enabled', True):
        cmd.append("wandb.enabled=true")
        cmd.append(f"wandb.project={wandb_config['project']}")
        if wandb_config.get('entity'):
            cmd.append(f"wandb.entity={wandb_config['entity']}")
    else:
        cmd.append("wandb.enabled=false")
    
    return cmd


def run_experiment(cmd: List[str], trial_num: int, total_trials: int, 
                   hyperparams: Dict[str, Any], dry_run: bool = False) -> bool:
    """
    Run a single experiment with given hyperparameters.
    
    Args:
        cmd: Command to execute
        trial_num: Current trial number
        total_trials: Total number of trials
        hyperparams: Hyperparameters for this trial
        dry_run: If True, only print command without executing
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"TRIAL {trial_num}/{total_trials}")
    print(f"{'='*80}")
    print("Hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    print()
    
    # Build command string
    if cmd[0].startswith("CUDA_VISIBLE_DEVICES="):
        env_var = cmd[0]
        cmd_str = env_var + " " + " ".join(cmd[1:])
    else:
        cmd_str = " ".join(cmd)
    
    print(f"Command: {cmd_str}")
    print()
    
    if dry_run:
        print("(Dry run - not executing)")
        return True
    
    # Execute command
    try:
        # Handle CUDA_VISIBLE_DEVICES environment variable
        if cmd[0].startswith("CUDA_VISIBLE_DEVICES="):
            import os
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
        
        print(f"\n✓ Trial {trial_num} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Trial {trial_num} failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for evaluation harness")
    parser.add_argument("--search-config", type=str, default="hyper_search_code_transformer.yaml",
                       help="Path to hyperparameter search config")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--continue-on-failure", action="store_true",
                       help="Continue search even if some trials fail")
    
    args = parser.parse_args()
    
    # Load search config
    print("Loading search configuration...")
    config = load_search_config(args.search_config)
    
    base_config = config['base_config']
    method_name = config['method_name']
    hyperparameters = config['hyperparameters']
    search_strategy = config.get('search_strategy', 'grid')
    max_trials = config.get('max_trials')
    random_seed = config.get('random_seed', 42)
    wandb_config = config.get('wandb', {})
    cuda_device = config.get('cuda_device')
    
    # Generate hyperparameter combinations
    print(f"\nGenerating hyperparameter combinations ({search_strategy} search)...")
    combinations = generate_combinations(
        hyperparameters,
        strategy=search_strategy,
        max_trials=max_trials,
        seed=random_seed
    )
    
    print(f"Total trials to run: {len(combinations)}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - Commands will not be executed ***\n")
    
    # Run experiments
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, hyperparams in enumerate(combinations, 1):
        cmd = build_command(
            base_config=base_config,
            method_name=method_name,
            hyperparams=hyperparams,
            wandb_config=wandb_config,
            cuda_device=cuda_device
        )
        
        success = run_experiment(
            cmd=cmd,
            trial_num=i,
            total_trials=len(combinations),
            hyperparams=hyperparams,
            dry_run=args.dry_run
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            if not args.continue_on_failure:
                print("\nStopping search due to failure")
                break
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total trials: {len(combinations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    if not args.dry_run:
        print(f"\nResults saved to WandB project: {wandb_config.get('project', 'N/A')}")


if __name__ == "__main__":
    main()
