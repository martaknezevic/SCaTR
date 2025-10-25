"""
Efficient storage module for choices and metrics data.
Uses compressed formats to minimize disk space.
"""
import os
import json
import pickle
import gzip
from typing import List, Dict, Any
from dataclasses import asdict
from filelock import FileLock


class ChoiceStorage:
    """Stores full choice objects efficiently"""
    
    @staticmethod
    def serialize_choice(choice: Any) -> Dict[str, Any]:
        """Serialize a choice object to a dictionary"""
        result = {
            'message': {
                'content': choice.message.content,
                'role': choice.message.role
            },
            'finish_reason': choice.finish_reason,
            'index': choice.index,
        }
        
        # Store logprobs if available
        if hasattr(choice, 'logprobs') and choice.logprobs:
            logprobs_data = []
            for token_data in choice.logprobs.content:
                token_info = {
                    'token': token_data.token,
                    'logprob': token_data.logprob,
                    'top_logprobs': []
                }
                
                if token_data.top_logprobs:
                    for top_lp in token_data.top_logprobs:
                        token_info['top_logprobs'].append({
                            'token': top_lp.token,
                            'logprob': top_lp.logprob,
                        })
                
                logprobs_data.append(token_info)
            
            result['logprobs'] = {
                'content': logprobs_data
            }
        
        return result
    
    @staticmethod
    def save_choices(
        problem_id: str,
        choices: List[Any],
        output_dir: str,
        use_compression: bool = True
    ):
        """Save choices to disk efficiently"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Serialize all choices
        serialized_choices = []
        for i, choice in enumerate(choices):
            choice_data = ChoiceStorage.serialize_choice(choice)
            choice_data['rollout_idx'] = i
            choice_data['problem_id'] = problem_id
            serialized_choices.append(choice_data)
        
        # Save with or without compression
        filename = os.path.join(output_dir, f"{problem_id}_choices.pkl")
        
        if use_compression:
            filename += ".gz"
            with gzip.open(filename, 'wb') as f:
                pickle.dump(serialized_choices, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(serialized_choices, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return filename
    
    @staticmethod
    def load_choices(filename: str) -> List[Dict[str, Any]]:
        """Load choices from disk"""
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                return pickle.load(f)


class MetricsStorage:
    """Stores metrics data efficiently"""
    
    @staticmethod
    def save_response_metrics(
        problem_id: str,
        rollout_idx: int,
        metrics_data: Dict[str, Any],
        output_file: str
    ):
        """
        Append metrics for a single response to a JSONL file.
        Each line is a complete record for one response.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        record = {
            'problem_id': problem_id,
            'rollout_idx': rollout_idx,
            **metrics_data
        }
        
        # Append to JSONL file
        with open(output_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    @staticmethod
    def save_all_metrics_batch(
        metrics_list: List[Dict[str, Any]],
        output_file: str,
        use_compression: bool = True
    ):
        """Safely append a batch of metrics (multi-process safe)"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        lock = FileLock(output_file + ".lock")

        with lock:  # ensures exclusive access
            if use_compression:
                output_file = output_file.replace('.jsonl', '.jsonl.gz')
                with gzip.open(output_file, 'at', encoding='utf-8') as f:
                    for record in metrics_list:
                        f.write(json.dumps(record) + '\n')
            else:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for record in metrics_list:
                        f.write(json.dumps(record) + '\n')

        return output_file
        
    @staticmethod
    def load_metrics(filename: str) -> List[Dict[str, Any]]:
        """Load metrics from JSONL file"""
        metrics = []
        
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                for line in f:
                    metrics.append(json.loads(line.strip()))
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    metrics.append(json.loads(line.strip()))
        
        return metrics


class ResultsAggregator:
    """Aggregates results across entire dataset"""
    
    @staticmethod
    def aggregate_strategy_results(
        problem_results: List[Dict[str, Any]],
        dataset_info: Dict[str, Any],
        output_file: str
    ):
        """
        Aggregate strategy-level results for entire dataset.
        One JSON file with high-level statistics.
        """
        strategy_stats = {}
        total_problems = len(problem_results)
        
        # Initialize strategy tracking
        for problem_result in problem_results:
            if 'strategies' not in problem_result:
                continue
            
            for strategy_name in problem_result['strategies'].keys():
                if strategy_name not in strategy_stats:
                    strategy_stats[strategy_name] = {
                        'correct': 0,
                        'total': 0,
                        'accuracy': 0.0
                    }
        
        # Count correct answers per strategy
        for problem_result in problem_results:
            if 'error' in problem_result:
                continue
            
            strategies = problem_result.get('strategies', {})
            for strategy_name, selected_result in strategies.items():
                if strategy_name in strategy_stats:
                    strategy_stats[strategy_name]['total'] += 1
                    if selected_result.is_correct:
                        strategy_stats[strategy_name]['correct'] += 1
        
        # Calculate accuracies
        for strategy_name in strategy_stats:
            stats = strategy_stats[strategy_name]
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
        
        # Find best strategy
        best_strategy = max(
            strategy_stats.items(),
            key=lambda x: x[1]['accuracy']
        ) if strategy_stats else (None, None)
        
        # Compile final results
        results = {
            'dataset_info': dataset_info,
            'strategy_results': strategy_stats,
            'best_strategy': {
                'name': best_strategy[0],
                'stats': best_strategy[1]
            } if best_strategy[0] else None,
            'total_problems': total_problems
        }
        
        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results