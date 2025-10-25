"""
Utilities for analyzing stored evaluation results.
"""
import json
import gzip
import os
import pandas as pd
from typing import List, Dict, Any
import numpy as np


class ResultsAnalyzer:
    """Analyze evaluation results from storage"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, 'all_response_metrics.jsonl')
        self.strategies_file = os.path.join(output_dir, 'strategy_results.json')
    
    def load_all_metrics(self) -> pd.DataFrame:
        """Load all response metrics into a DataFrame"""
        metrics = []
        
        if self.metrics_file.endswith('.gz'):
            with gzip.open(self.metrics_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    metrics.append(json.loads(line.strip()))
        else:
            metrics_file_alt = self.metrics_file.replace('.gz', '')
            if os.path.exists(metrics_file_alt):
                with open(metrics_file_alt, 'r', encoding='utf-8') as f:
                    for line in f:
                        metrics.append(json.loads(line.strip()))
        
        return pd.DataFrame(metrics)
    
    def load_strategy_results(self) -> Dict[str, Any]:
        """Load strategy-level results"""
        with open(self.strategies_file, 'r') as f:
            return json.load(f)
    
    def analyze_metric_distributions(self, metric_name: str) -> Dict[str, Any]:
        """Analyze distribution of a specific metric"""
        df = self.load_all_metrics()
        
        # Filter metric columns
        metric_cols = [col for col in df.columns if metric_name in col]
        
        stats = {}
        for col in metric_cols:
            if col in df.columns:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }
        
        return stats
    
    def compare_correct_vs_incorrect(self, metric_name: str) -> Dict[str, Any]:
        """Compare metric values between correct and incorrect responses"""
        df = self.load_all_metrics()
        
        correct = df[df['is_correct'] == True]
        incorrect = df[df['is_correct'] == False]
        
        metric_cols = [col for col in df.columns if metric_name in col]
        
        comparison = {}
        for col in metric_cols:
            if col in df.columns:
                comparison[col] = {
                    'correct': {
                        'mean': float(correct[col].mean()) if len(correct) > 0 else None,
                        'std': float(correct[col].std()) if len(correct) > 0 else None,
                        'count': len(correct)
                    },
                    'incorrect': {
                        'mean': float(incorrect[col].mean()) if len(incorrect) > 0 else None,
                        'std': float(incorrect[col].std()) if len(incorrect) > 0 else None,
                        'count': len(incorrect)
                    }
                }
        
        return comparison
    
    def get_problem_difficulty(self) -> pd.DataFrame:
        """Analyze problem difficulty based on success rate"""
        df = self.load_all_metrics()
        
        difficulty = df.groupby('problem_id').agg({
            'is_correct': ['sum', 'count', 'mean']
        }).reset_index()
        
        difficulty.columns = ['problem_id', 'correct_count', 'total_attempts', 'success_rate']
        difficulty = difficulty.sort_values('success_rate')
        
        return difficulty
    
    def get_best_metric_for_selection(self) -> Dict[str, float]:
        """Find which metric aggregation works best for selection"""
        strategy_results = self.load_strategy_results()
        
        # Extract accuracy for each strategy
        strategies = strategy_results.get('strategy_results', {})
        
        # Group by metric type
        metric_accuracies = {}
        for strategy_name, stats in strategies.items():
            if strategy_name in ['random', 'oracle']:
                continue
            
            accuracy = stats['accuracy']
            
            # Extract metric name from strategy
            if 'highest_' in strategy_name:
                metric = strategy_name.replace('highest_', '')
            elif 'lowest_' in strategy_name:
                metric = strategy_name.replace('lowest_', '')
            else:
                continue
            
            if metric not in metric_accuracies:
                metric_accuracies[metric] = {'highest': None, 'lowest': None}
            
            if 'highest_' in strategy_name:
                metric_accuracies[metric]['highest'] = accuracy
            else:
                metric_accuracies[metric]['lowest'] = accuracy
        
        return metric_accuracies
    
    def generate_summary_report(self, output_file: str = None):
        """Generate comprehensive summary report"""
        strategy_results = self.load_strategy_results()
        df = self.load_all_metrics()
        
        report = {
            'dataset_info': strategy_results['dataset_info'],
            'overall_statistics': {
                'total_responses': len(df),
                'total_correct': int(df['is_correct'].sum()),
                'overall_accuracy': float(df['is_correct'].mean()),
                'unique_problems': df['problem_id'].nunique()
            },
            'strategy_performance': strategy_results['strategy_results'],
            'best_strategy': strategy_results['best_strategy'],
            'problem_difficulty': self.get_problem_difficulty().to_dict('records'),
            'metric_effectiveness': self.get_best_metric_for_selection()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self):
        """Print human-readable summary"""
        report = self.generate_summary_report()
        
        print("="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print("\nDataset Information:")
        for key, value in report['dataset_info'].items():
            print(f"  {key}: {value}")
        
        print("\nOverall Statistics:")
        for key, value in report['overall_statistics'].items():
            if 'accuracy' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        
        print("\nTop 5 Best Strategies:")
        strategies = sorted(
            report['strategy_performance'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )[:5]
        
        for strategy_name, stats in strategies:
            print(f"  {strategy_name}:")
            print(f"    Accuracy: {stats['accuracy']:.2%}")
            print(f"    Correct: {stats['correct']}/{stats['total']}")
        
        print("\nMost Difficult Problems (lowest success rate):")
        difficult_problems = sorted(
            report['problem_difficulty'],
            key=lambda x: x['success_rate']
        )[:5]
        
        for prob in difficult_problems:
            print(f"  {prob['problem_id']}: {prob['success_rate']:.2%} " 
                  f"({prob['correct_count']}/{prob['total_attempts']})")
        
        print("="*80)


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <output_dir>")
        print("Example: python analysis.py ./outputs/MathArena_aime_2025/turn1")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    print(output_dir)
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist")
        sys.exit(1)
    
    analyzer = ResultsAnalyzer(output_dir)
    
    print("Loading results...")
    analyzer.print_summary()
    
    # Generate detailed report
    report_file = os.path.join(output_dir, 'detailed_analysis.json')
    analyzer.generate_summary_report(report_file)
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()