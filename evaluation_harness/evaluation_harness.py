from config_schema import EvaluationConfig, MethodConfig
from evaluator import CodeEvaluator, Evaluator, MathEvaluator
from data_utils import load_data_parallel, split_data
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import wandb
from methods.method_registry import get_global_registry

class EvaluationHarness:
    """Main orchestrator for evaluation pipeline - initializes and coordinates everything"""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation harness
        
        Args:
            config: Evaluation configuration object
        """
        # Store config
        # Initialize data loaders (once for all methods) TODO
        # Initialize evaluator (once for all methods)
        self.config = config
        
        self.setup()
        
        #if config.evaluation == 'math':
        #    self.evaluator = MathEvaluator(None) # TODO pass test dataset
        #elif config.evaluation == 'code':
        #    self.evaluator = CodeEvaluator(None) # TODO pass test dataset
            
        self.registry = get_global_registry()

    
    def setup(self):
        """
        Setup evaluation environment and load data
        
        This method:
        1. Creates DataLoader for train dataset (if specified)
        2. Creates DataLoader for validation dataset (if specified)
        3. Creates DataLoader for test dataset
        4. Initializes Evaluator with test dataset
        5. Creates output directories
        """
        
        if self.config.data.train_data_path:
            # load train data
            # split, train/val/test data
            # initialize data loaders for all data splits
            # Load data
            print("Step 1: Loading train data...")
            print("-" * 80)
            train_choices_dir = Path(os.path.join(self.config.data.train_data_path, "choices"))
            
            train_data_name = self.config.data.train_data_path.replace("/", "_")

            all_train_data, train_metrics = load_data_parallel(
                choices_dir=train_choices_dir,
                metrics_file=Path(os.path.join(self.config.data.train_data_path, "all_response_metrics.jsonl")),
                dataset_name=train_data_name,
                num_problems=self.config.data.num_problems,
                num_workers=self.config.data.num_workers,
                num_choice_workers=self.config.data.num_choice_workers,
                use_cache=self.config.data.use_cache,
                subsample_size=self.config.data.subsample_size,
                cache_dir=Path(self.config.data.cache_dir) if self.config.data.cache_dir else None
            )
            
            # Split data into train/val/test
            print("\nStep 2: Splitting data into train/val/test...")
            print("-" * 80)
            train_data, val_data, test_data = split_data(
                all_train_data,
                num_train_problems=self.config.data.num_train_problems,
                num_val_problems=self.config.data.num_val_problems,
                num_test_problems=self.config.data.num_test_problems,
                random_seed=self.config.data.random_seed
            )

            print(f"{len(train_data)} train rollouts")
            print(f"{len(val_data)} validation rollouts")
            print(f"{len(test_data)} test rollouts")
            self.train_data = train_data
            self.val_data = val_data
            self.test_data = test_data
        else:
            self.train_data = None
            self.val_data = None
            self.test_data = None

        # Load external test data
        print("\nStep 3: Loading external test data...")
        print("-" * 80)
        external_test_paths = self.config.data.external_test_data_paths
        if isinstance(external_test_paths, str):
            external_test_paths = [external_test_paths] if external_test_paths else []
        
        self.external_test_data = {}
        for test_path in external_test_paths:
            if not test_path:
                continue
            test_choices_dir = Path(os.path.join(test_path, "choices"))
            test_data_name = test_path.replace("/", "_")
            test_data, test_metrics = load_data_parallel(
                choices_dir=test_choices_dir,
                metrics_file=Path(os.path.join(test_path, "all_response_metrics.jsonl")),
                dataset_name=test_data_name,
                num_problems=self.config.data.num_problems,
                num_workers=self.config.data.num_workers,
                num_choice_workers=self.config.data.num_choice_workers,
                use_cache=self.config.data.use_cache,
                subsample_size=self.config.data.subsample_size,
                cache_dir=Path(self.config.data.cache_dir) if self.config.data.cache_dir else None
            )
            self.external_test_data[test_data_name] = {
                'data': test_data,
                'path': test_path
            }
            print(f"Loaded {len(test_data)} rollouts from {test_path}")
            
        # Setup output directory
        self.output_dir = Path(self.config.output.base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config to run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
            
        print(f"\n{'='*80}")
        print(f"Evaluation Harness v2 - Run Directory: {self.run_dir}")
        print(f"{'='*80}\n")
    
    def run_method_evaluation(self, method_config: MethodConfig) -> Dict[str, Any]:
        """
        Run evaluation for a single method.
        
        This method:
        1. Gets method instance from registry using method_config
        2. Calls method.prepare_data() with loaded train/val/test data
        3. Calls method.train() if training data exists
        4. Calls method.predict_confidence() on test data
        5. Saves predictions to JSONL file
        6. Calls evaluator.evaluate() on predictions
        7. Returns evaluation results
        
        Args:
            method_config: Configuration for the method
            
        Returns:
            Dictionary containing evaluation results (accuracy, etc.)
        """
        method_name = method_config.name
        method_params = method_config.params
        
        print(f"\n{'='*80}")
        print(f"METHOD: {method_name.upper()}")
        print(f"Parameters: {method_params}")
        print(f"{'='*80}")
        
        # Create method instance
        common_params = {
            'tail_tokens': self.config.data.tail_tokens,
            'device': self.config.device,
            'seed': self.config.data.random_seed,
            'method_name': method_name,
            'train_data_path': self.config.data.train_data_path,
        }
        
        # Merge params
        full_config = {**common_params, **method_params}
        
        method = self.registry.create(
            method_name,
            method_params,
            common_params=common_params
        )
        
        # Set seed for reproducibility
        method.set_seed(self.config.data.random_seed)
        
        # Create method-specific output directory
        method_id = method.get_method_id()
        method_dir = self.run_dir / method_id
        method_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Prepare data (method-specific preprocessing)
        print("\nStep 1: Preparing data for method...")
        train_data = self.train_data if hasattr(self, 'train_data') else None
        val_data = self.val_data if hasattr(self, 'val_data') else None
        test_data = self.test_data if hasattr(self, 'test_data') else None
        
        prepared_train, prepared_val, prepared_test = method.prepare_data(
            train_data, val_data, test_data
        )
        
        # Step 2: Train method (if train data provided)
        train_metrics = {}
        if prepared_train is not None and len(prepared_train) > 0:
            print(f"\nStep 2: Training {method_name}...")
            train_metrics = method.train(
                prepared_train,
                prepared_val
            )
            print(f"Training metrics: {train_metrics}")
            
            # Log training metrics to wandb
            if self.config.wandb.enabled and train_metrics:
                wandb.log({f"train/{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))})
            
            # Save model if method supports it
            if self.config.output.save_models:
                model_path = method.save(str(method_dir))
                if model_path:
                    print(f"Model saved to: {model_path}")
        else:
            print(f"\nStep 2: No training data, skipping training...")
        
        # Step 3: Evaluate on train/val/test splits
        print(f"\nStep 3: Evaluating on train/val/test splits...")
        train_results = None
        val_results = None
        test_results = None
        
        if prepared_train is not None and len(prepared_train) > 0:
            train_results = method.evaluate(prepared_train)
            print(f"Train Accuracy: {train_results['accuracy']:.2%} "
                  f"({train_results['correct_count']}/{train_results['total_count']})")
            
            # Log train results to wandb
            if self.config.wandb.enabled:
                wandb.log({
                    'final_train_accuracy': train_results['accuracy'],
                    'final_train_correct': train_results['correct_count'],
                    'final_train_total': train_results['total_count']
                })
        
        if prepared_val is not None and len(prepared_val) > 0:
            val_results = method.evaluate(prepared_val)
            print(f"Val Accuracy: {val_results['accuracy']:.2%} "
                  f"({val_results['correct_count']}/{val_results['total_count']})")
            
            # Log val results to wandb
            if self.config.wandb.enabled:
                wandb.log({
                    'final_val_accuracy': val_results['accuracy'],
                    'final_val_correct': val_results['correct_count'],
                    'final_val_total': val_results['total_count']
                })
        
        if prepared_test is not None and len(prepared_test) > 0:
            test_results = method.evaluate(prepared_test)
            print(f"Test Accuracy (internal): {test_results['accuracy']:.2%} "
                  f"({test_results['correct_count']}/{test_results['total_count']})")
            
            # Log test results to wandb
            if self.config.wandb.enabled:
                wandb.log({
                    'final_test_accuracy': test_results['accuracy'],
                    'final_test_correct': test_results['correct_count'],
                    'final_test_total': test_results['total_count']
                })
            
            # Save internal test predictions
            if self.config.output.save_predictions:
                test_predictions = method.format_output(prepared_test)
                predictions_file = method_dir / "predictions_internal_test.jsonl"
                method.save_predictions(test_predictions, str(predictions_file))
                print(f"Internal test predictions saved to: {predictions_file}")
        
        # Step 4: Evaluate on external test datasets
        print(f"\nStep 4: Evaluating on external test datasets...")
        external_test_results = {}
        
        for test_name, test_info in self.external_test_data.items():
            print(f"\nEvaluating on {test_name}...")
            external_data = test_info['data']
            external_path = test_info['path']
            
            # Prepare external test data
            _, _, prepared_external = method.prepare_data(None, None, external_data)
            
            # For neural_net method, switch to external test metrics file
            if method_name == 'neural_net':
                method.metrics_file = Path(os.path.join(external_path, 'all_response_metrics.jsonl'))
            
            # Evaluate on external test set
            external_results = method.evaluate(prepared_external)
            
            # Switch back to train metrics file
            if method_name == 'neural_net' and self.config.data.train_data_path:
                method.metrics_file = Path(os.path.join(self.config.data.train_data_path, 'all_response_metrics.jsonl'))
            
            print(f"{test_name} Accuracy: {external_results['accuracy']:.2%} "
                  f"({external_results['correct_count']}/{external_results['total_count']})")
            
            external_test_results[test_name] = external_results
            
            # Log external test results to wandb
            if self.config.wandb.enabled:
                wandb.log({
                    f'external_{test_name}_accuracy': external_results['accuracy'],
                    f'external_{test_name}_correct': external_results['correct_count'],
                    f'external_{test_name}_total': external_results['total_count']
                })
            
            # Save external test predictions
            if self.config.output.save_predictions:
                external_predictions = method.format_output(prepared_external)
                predictions_file = method_dir / f"predictions_{test_name}.jsonl"
                method.save_predictions(external_predictions, str(predictions_file))
                print(f"Predictions saved to: {predictions_file}")
        
        # Step 5: Save metrics
        if self.config.output.save_metrics:
            metrics_file = method_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    'train_metrics': train_metrics,
                    'train_results': train_results,
                    'val_results': val_results,
                    'test_results': test_results,
                    'external_test_results': external_test_results,
                    'method_config': method_params,
                }, f, indent=2)
            print(f"Metrics saved to: {metrics_file}")
        
        return {
            'method_id': method_id,
            'method_name': method_name,
            'method_params': method_params,
            'train_metrics': train_metrics,
            'train_results': train_results,
            'val_results': val_results,
            'test_results': test_results,
            'external_test_results': external_test_results,
        }
    
    def run_all_evaluations(self) -> Dict[str, Any]:
        """
        Run evaluations for all configured methods.
        
        This method:
        1. Iterates through all method configs
        2. Calls run_method_evaluation() for each
        3. Collects all results
        4. Generates summary report
        
        Returns:
            Dictionary mapping method IDs to results
        """
        all_results = {
            'config': self.config.to_dict(),
            'methods': {}
        }
        
        print(f"\n{'='*80}")
        print(f"Running {len(self.config.methods)} method(s)")
        print(f"{'='*80}")
        
        for i, method_config in enumerate(self.config.methods, 1):
            print(f"\n[{i}/{len(self.config.methods)}] Evaluating method: {method_config.name}")
            
            try:
                result = self.run_method_evaluation(method_config)
                
                # Store results with unique key
                method_id = result['method_id']
                # Handle duplicate method IDs
                original_id = method_id
                counter = 1
                while method_id in all_results['methods']:
                    method_id = f"{original_id}_{counter}"
                    counter += 1
                
                all_results['methods'][method_id] = result
                
            except Exception as e:
                print(f"\n❌ ERROR evaluating {method_config.name}: {e}")
                import traceback
                traceback.print_exc()
                
                all_results['methods'][method_config.name] = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        return all_results

    
    def generate_summary_report(self, all_results: Dict[str, Any]) -> str:
        """
        Generate summary report of all evaluations.
        
        Creates:
        1. Summary JSON file with all results
        2. Summary table showing accuracy across methods
        3. Detailed report text file
        
        Args:
            all_results: Dictionary of all method results
            
        Returns:
            Path to generated summary report
        """
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*80}")
        
        # Save complete results JSON
        results_file = self.run_dir / 'all_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Complete results saved to: {results_file}")
        
        # Generate summary tables for each split
        for split in ['train', 'val', 'test']:
            self._generate_summary_table(all_results, split=split)
        
        # Generate summary tables for all external test datasets
        self._generate_external_test_tables(all_results)
        
        # Generate detailed report
        report_file = self.run_dir / 'summary_report.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION HARNESS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Train Dataset: {self.config.data.train_data_path or 'None'}\n")
            external_paths = self.config.data.external_test_data_paths
            if isinstance(external_paths, list):
                f.write(f"External Test Datasets: {', '.join(external_paths)}\n")
            else:
                f.write(f"External Test Datasets: {external_paths or 'None'}\n")
            f.write(f"Evaluation Type: {self.config.evaluation}\n")
            f.write(f"Device: {self.config.device}\n")
            f.write(f"Random Seed: {self.config.data.random_seed}\n")
            f.write(f"Number of Methods: {len(self.config.methods)}\n")
            f.write("\n")
            
            # Results for each method
            f.write("METHOD RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            for method_id, result in all_results['methods'].items():
                f.write(f"Method: {method_id}\n")
                
                if 'error' in result:
                    f.write(f"  ERROR: {result['error']}\n\n")
                    continue
                
                # Test results
                test_res = result.get('test_results', {})
                if test_res:
                    acc = test_res.get('accuracy', 0)
                    correct = test_res.get('correct_count', 0)
                    total = test_res.get('total_count', 0)
                    f.write(f"  Test Accuracy: {acc:.2%} ({correct}/{total})\n")
                
                # Train results
                train_res = result.get('train_results', {})
                if train_res:
                    acc = train_res.get('accuracy', 0)
                    correct = train_res.get('correct_count', 0)
                    total = train_res.get('total_count', 0)
                    f.write(f"  Train Accuracy: {acc:.2%} ({correct}/{total})\n")
                
                # Val results
                val_res = result.get('val_results', {})
                if val_res:
                    acc = val_res.get('accuracy', 0)
                    correct = val_res.get('correct_count', 0)
                    total = val_res.get('total_count', 0)
                    f.write(f"  Val Accuracy: {acc:.2%} ({correct}/{total})\n")
                
                # External test results
                external_results = result.get('external_test_results', {})
                if external_results:
                    f.write("\n  External Test Results:\n")
                    for test_name, ext_res in external_results.items():
                        acc = ext_res.get('accuracy', 0)
                        correct = ext_res.get('correct_count', 0)
                        total = ext_res.get('total_count', 0)
                        f.write(f"    {test_name}: {acc:.2%} ({correct}/{total})\n")
                
                f.write("\n")
        
        print(f"\n✓ Summary report saved to: {report_file}")
        
        return str(report_file)
    
    def _print_method_results(self, method_name: str, train_results: Dict,
                            val_results: Optional[Dict] = None,
                            test_results: Optional[Dict] = None):
        """Print formatted results for a method."""
        print(f"\n{method_name.upper()} RESULTS")
        print("-" * 80)

        # Train results
        if 'std_accuracy' in train_results:  # Random method
            print(f"Train: {train_results['accuracy']:.2%} ± {train_results['std_accuracy']:.2%} "
                  f"({train_results['correct_count']}/{train_results['total_count']})")
        else:
            print(f"Train: {train_results['accuracy']:.2%} "
                  f"({train_results['correct_count']}/{train_results['total_count']})")

        # Val results
        if val_results:
            if 'std_accuracy' in val_results:
                print(f"Val:   {val_results['accuracy']:.2%} ± {val_results['std_accuracy']:.2%} "
                      f"({val_results['correct_count']}/{val_results['total_count']})")
            else:
                print(f"Val:   {val_results['accuracy']:.2%} "
                      f"({val_results['correct_count']}/{val_results['total_count']})")

        # Test results
        if test_results:
            if 'std_accuracy' in test_results:
                print(f"Test:  {test_results['accuracy']:.2%} ± {test_results['std_accuracy']:.2%} "
                      f"({test_results['correct_count']}/{test_results['total_count']})")
            else:
                print(f"Test:  {test_results['accuracy']:.2%} "
                      f"({test_results['correct_count']}/{test_results['total_count']})")

    def _generate_summary_table(self, all_results: Dict, split='test'):
        """Generate summary table showing accuracy across methods."""
        
        # Extract: {method: (acc, correct, total)}
        table_data = {}

        for method_name, method_results in all_results.get("methods", {}).items():

            # Handle errors
            if "error" in method_results:
                table_data[method_name] = (None, None, None)
                continue

            key = f"{split}_results"
            results = method_results.get(key)

            if results:
                acc     = float(results.get("accuracy", None))
                correct = results.get("correct_count", None)
                total   = results.get("total_count", None)
            else:
                acc = correct = total = None

            table_data[method_name] = (acc, correct, total)

        if not table_data:
            print(f"No {split} results available.")
            return

        print("table_data =", table_data)

        # List of methods
        methods = sorted(table_data.keys())

        # Build table
        lines = []
        lines.append("\n" + "="*80)
        lines.append(f"SUMMARY TABLE: {split.capitalize()} Set Accuracy")
        lines.append("="*80)
        lines.append("")

        # Column widths
        col_width = max(20, max(len(m) for m in methods))

        # Header
        header = ""
        for m in methods:
            header += f" | {m:^{col_width}}"
        lines.append(header)

        # Separator
        sep = ""
        for _ in methods:
            sep += "-+-" + "-" * col_width
        lines.append(sep)

        # Single row (because there is no dataset dimension)
        row = ""
        for m in methods:
            acc, correct, total = table_data[m]
            if acc is not None:
                value = f"{acc*100:.2f}% ({correct}/{total})"
            else:
                value = "N/A"
            row += f" | {value:^{col_width}}"
        lines.append(row)

        lines.append("")

        table_text = "\n".join(lines)
        print(table_text)

        # Save
        summary_file = self.run_dir / f"summary_table_{split}.txt"
        with open(summary_file, "w") as f:
            f.write(table_text)

        print(f"Summary table saved to: {summary_file}")

    def _generate_external_test_tables(self, all_results: Dict):
        """Generate summary tables for all external test datasets."""
        
        # Collect all external test dataset names
        external_test_names = set()
        for method_name, method_results in all_results.get("methods", {}).items():
            if "error" not in method_results:
                external_results = method_results.get('external_test_results', {})
                external_test_names.update(external_results.keys())
        
        if not external_test_names:
            print("\nNo external test datasets found.")
            return
        
        # Generate a table for each external test dataset
        for test_name in sorted(external_test_names):
            self._generate_external_test_table(all_results, test_name)
    
    def _generate_external_test_table(self, all_results: Dict, test_name: str):
        """Generate summary table for a specific external test dataset."""
        
        # Extract: {method: (acc, correct, total)}
        table_data = {}

        for method_name, method_results in all_results.get("methods", {}).items():
            # Handle errors
            if "error" in method_results:
                table_data[method_name] = (None, None, None)
                continue

            external_results = method_results.get('external_test_results', {})
            results = external_results.get(test_name)

            if results:
                acc = float(results.get("accuracy", 0))
                correct = results.get("correct_count", None)
                total = results.get("total_count", None)
            else:
                acc = correct = total = None

            table_data[method_name] = (acc, correct, total)

        if not table_data:
            print(f"No results available for external test: {test_name}")
            return

        # List of methods
        methods = sorted(table_data.keys())

        # Build table
        lines = []
        lines.append("\n" + "="*80)
        lines.append(f"SUMMARY TABLE: External Test - {test_name}")
        lines.append("="*80)
        lines.append("")

        # Column widths
        col_width = max(20, max(len(m) for m in methods))

        # Header
        header = ""
        for m in methods:
            header += f" | {m:^{col_width}}"
        lines.append(header)

        # Separator
        sep = ""
        for _ in methods:
            sep += "-+-" + "-" * col_width
        lines.append(sep)

        # Single row
        row = ""
        for m in methods:
            acc, correct, total = table_data[m]
            if acc is not None:
                value = f"{acc*100:.2f}% ({correct}/{total})"
            else:
                value = "N/A"
            row += f" | {value:^{col_width}}"
        lines.append(row)

        lines.append("")

        table_text = "\n".join(lines)
        print(table_text)

        # Save
        summary_file = self.run_dir / f"summary_table_external_{test_name.replace('/', '_')}.txt"
        with open(summary_file, "w") as f:
            f.write(table_text)

        print(f"Summary table saved to: {summary_file}")