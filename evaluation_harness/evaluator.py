from abc import ABC, abstractmethod
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class Evaluator(ABC):
    """Base class for evaluating method predictions"""
    
    def __init__(self, evaluation_type: str = 'math'):
        """
        Initialize evaluator
        
        Args:
            evaluation_type: Type of evaluation ('math' or 'code')
        """
        self.evaluation_type = evaluation_type
    
    @abstractmethod
    def evaluate(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth
        
        Args:
            predictions: List of prediction dictionaries from predictions.jsonl
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @staticmethod
    def load_predictions(prediction_file: str) -> List[Dict]:
        """
        Load predictions from JSONL file
        
        Args:
            prediction_file: Path to prediction JSONL file
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        with open(prediction_file, 'r') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        return predictions
    
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to file
        
        Args:
            results: Dictionary of evaluation results
            output_path: Path to save results
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    
class MathEvaluator(Evaluator):
    """Evaluator for mathematical problem datasets"""
    
    def __init__(self):
        """Initialize MathEvaluator"""
        super().__init__(evaluation_type='math')
        
    def _extract_answer_from_response(self, response_text: str) -> Optional[int]:
        """
        Extract numerical answer from response text.
        
        Supports AIME-style answers (integers from 0 to 999).
        Can be extended for other math formats.
        
        Args:
            response_text: Response text containing the answer
            
        Returns:
            Extracted answer as integer, or None if extraction failed
        """
        if not response_text:
            return None
        
        # Look for patterns like "answer is 123", "= 123", "123.", etc.
        patterns = [
            r'(?:answer|result|solution)(?:\s+is)?\s*[:\=]?\s*(\d{1,3})',
            r'(?:^|\s)(\d{1,3})(?:\s*$|\.$)',
            r'(?:equals?|is)\s+(\d{1,3})',
            r'(\d{1,3})(?:\s*$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text.lower(), re.MULTILINE)
            if matches:
                answer = int(matches[-1])  # Take the last match
                if 0 <= answer <= 999:  # AIME constraint
                    return answer
        
        # If no clear answer found, return None
        return None
    
    def evaluate(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate math problem predictions.
        
        Expects each prediction to have:
        - 'is_correct': boolean indicating if the answer is correct
        - 'problem_id': identifier for the problem
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary with accuracy and counts
        """
        correct = 0
        total = len(predictions)
        
        problem_results = {}
        
        for pred in predictions:
            problem_id = pred.get('problem_id', 'unknown')
            is_correct = pred.get('is_correct', False)
            
            if is_correct:
                correct += 1
            
            problem_results[problem_id] = {
                'is_correct': is_correct,
                'confidence': pred.get('confidence', 0.0),
                'answer': pred.get('answer', '')
            }
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_count': correct,
            'total_count': total,
            'problem_results': problem_results
        }
    
    
class CodeEvaluator(Evaluator):
    """Evaluator for code generation datasets"""
    
    def __init__(self):
        """Initialize CodeEvaluator"""
        super().__init__(evaluation_type='code')
    
    def evaluate(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate code generation predictions.
        
        For code tasks, typically checks if generated code passes test cases.
        Expects 'is_correct' field in predictions indicating if code passed tests.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary with pass@1 accuracy and counts
        """
        correct = 0
        total = len(predictions)
        
        problem_results = {}
        
        for pred in predictions:
            problem_id = pred.get('problem_id', 'unknown')
            is_correct = pred.get('is_correct', False)
            
            if is_correct:
                correct += 1
            
            problem_results[problem_id] = {
                'is_correct': is_correct,
                'confidence': pred.get('confidence', 0.0),
                'answer': pred.get('answer', '')
            }
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,  # Also called pass@1 for code tasks
            'correct_count': correct,
            'total_count': total,
            'problem_results': problem_results
        }


def get_evaluator(evaluation_type: str) -> Evaluator:
    """
    Factory function to get appropriate evaluator.
    
    Args:
        evaluation_type: Type of evaluation ('math' or 'code')
        
    Returns:
        Evaluator instance
    """
    if evaluation_type == 'math':
        return MathEvaluator()
    elif evaluation_type == 'code':
        return CodeEvaluator()
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}. Must be 'math' or 'code'")