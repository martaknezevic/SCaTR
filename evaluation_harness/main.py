"""
Main entry point for the evaluation harness.

Usage:
    # Basic usage with config file
    python main.py --config-name=config_code
    
    # With Hydra-style overrides (recommended for hyperparameter tuning)
    python main.py --config-name=config_kodcode_hyps methods.0.params.lr=0.001
    python main.py --config-name=config_kodcode_hyps methods.0.params.dropout=0.2 methods.0.params.weight_decay=0.0001
    
    # Multiple overrides
    python main.py --config-name=config_kodcode_hyps \\
        methods.0.params.lr=0.005 \\
        methods.0.params.dropout=0.1 \\
        methods.0.params.weight_decay=0.0001 \\
        methods.0.params.loss_type=hinge \\
        wandb.enabled=true \\
        output.base_dir=outputs/exp_001
"""

import sys
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set WandB API key from environment variable if available
if 'WANDB_API_KEY' in os.environ:
    wandb.login(key=os.environ['WANDB_API_KEY'])
elif Path.home().joinpath('.wandb_api_key').exists():
    # Alternative: read from ~/.wandb_api_key file
    with open(Path.home().joinpath('.wandb_api_key'), 'r') as f:
        api_key = f.read().strip()
        wandb.login(key=api_key)

from evaluation_harness import EvaluationHarness
from config_schema import EvaluationConfig
from methods.method_registry import get_global_registry


@hydra.main(version_base=None, config_path="/home/ubuntu/cactts/evaluation_harness/config", config_name="config_code")
def main(cfg: DictConfig) -> int:
    """
    Main entry point for running evaluations.
    
    Hydra automatically:
    - Loads the config from config_path/config_name.yaml
    - Applies command-line overrides (e.g., methods.0.params.lr=0.001)
    - Provides the merged config as cfg (DictConfig)
    
    Steps:
    1. Convert Hydra config to our config schema
    2. Initialize method registry
    3. Initialize evaluation harness
    4. Run all evaluations
    5. Generate summary report
    """
    print(f"{'='*80}")
    print(f"EVALUATION HARNESS")
    print(f"{'='*80}")
    print(f"Configuration loaded with Hydra overrides applied")
    
    try:
        # Convert OmegaConf to dict and create config object
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config = EvaluationConfig.from_dict(config_dict)
        
        # Validate configuration
        config.validate()
        
        # Initialize WandB if enabled
        if config.wandb.enabled:
            print("\nInitializing Weights & Biases...")
            
            # Log the entire config dictionary to wandb
            full_config = OmegaConf.to_container(cfg, resolve=True)
            
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name,
                tags=config.wandb.tags,
                config=full_config  # Log entire config
            )
            print(f"✓ WandB initialized: {config.wandb.project}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Initialize method registry and auto-discover methods
    print("\nInitializing method registry...")
    registry = get_global_registry()
    registry.auto_discover_methods("methods")
    print(f"✓ Registered methods: {', '.join(registry.list_methods())}")
    
    # Initialize evaluation harness with config
    print("\nInitializing evaluation harness...")
    try:
        harness = EvaluationHarness(config)
    except Exception as e:
        print(f"❌ Error initializing harness: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run evaluations
    print("\nStarting evaluations...")
    try:
        all_results = harness.run_all_evaluations()
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate summary report
    print("\nGenerating summary report...")
    try:
        report_path = harness.generate_summary_report(all_results)
        print(f"\n✓ Evaluation complete! Summary report: {report_path}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Log final summary to wandb
    if config.wandb.enabled:
        # Create a summary of all results
        summary = {}
        for method_id, method_result in all_results.get('methods', {}).items():
            if 'error' not in method_result:
                # Log final metrics
                if method_result.get('train_results'):
                    summary[f'{method_id}_final_train_acc'] = method_result['train_results']['accuracy']
                if method_result.get('val_results'):
                    summary[f'{method_id}_final_val_acc'] = method_result['val_results']['accuracy']
                if method_result.get('test_results'):
                    summary[f'{method_id}_final_test_acc'] = method_result['test_results']['accuracy']
                
                # Log external test results
                for ext_name, ext_result in method_result.get('external_test_results', {}).items():
                    summary[f'{method_id}_external_{ext_name}_acc'] = ext_result['accuracy']
        
        wandb.summary.update(summary)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    
    # Finish WandB run
    if config.wandb.enabled:
        wandb.finish()
    
    return 0


if __name__ == "__main__":
    main()