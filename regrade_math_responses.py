"""
Re-grading Script for Math Responses

This script re-grades all responses in all_response_metrics.jsonl files using
math answer extractors and graders. It loads problems from the original dataset,
maps them to responses by problem_id, extracts answers, and re-grades each response.

Usage:
    python regrade_math_responses.py --dataset datasets/aime.jsonl --metrics path/to/all_response_metrics.jsonl --loader aime
    python regrade_math_responses.py --dataset datasets/math.jsonl --metrics path/to/all_response_metrics.jsonl --loader math
    python regrade_math_responses.py --dataset datasets/gsm8k_1000.jsonl --metrics path/to/all_response_metrics.jsonl --loader gsm8k
"""

import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

# Import components from math_evaluator
from math_evaluator import (
    AIMELoader,
    MATHLoader,
    GSM8KLoader,
    GenericLoader,
    AIMEExtractor,
    MATHExtractor,
    GSM8KExtractor,
    GenericExtractor,
    ExactMatchGrader,
    MathGrader,
    MathProblem,
)


class LoaderFactory:
    """Factory for creating dataset loaders"""
    
    _loaders = {
        'aime': AIMELoader,
        'math': MATHLoader,
        'gsm8k': GSM8KLoader,
        'generic': GenericLoader,
    }
    
    @classmethod
    def create(cls, loader_type: str):
        """Create a loader instance by type"""
        loader_class = cls._loaders.get(loader_type.lower())
        if loader_class is None:
            available = ', '.join(cls._loaders.keys())
            raise ValueError(f"Unknown loader type: {loader_type}. Available: {available}")
        return loader_class()


class ExtractorFactory:
    """Factory for creating answer extractors"""
    
    _extractors = {
        'aime': AIMEExtractor,
        'math': MATHExtractor,
        'gsm8k': GSM8KExtractor,
        'generic': GenericExtractor,
    }
    
    @classmethod
    def create(cls, extractor_type: str):
        """Create an extractor instance by type"""
        extractor_class = cls._extractors.get(extractor_type.lower())
        if extractor_class is None:
            available = ', '.join(cls._extractors.keys())
            raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {available}")
        return extractor_class()


class MathResponseRegrader:
    """Re-grades math responses in metrics JSONL files"""
    
    def __init__(
        self,
        dataset_path: str,
        loader_type: str = "aime",
        extractor_type: Optional[str] = None,
        grader_type: str = "math",
        split: str = "test",
    ):
        """
        Initialize the math regrader
        
        Args:
            dataset_path: Path to the original dataset (JSONL file with problems)
            loader_type: Type of loader ('aime', 'math', or 'gsm8k')
            extractor_type: Type of extractor (defaults to same as loader_type)
            grader_type: Type of grader ('exact' or 'math')
            split: Dataset split to use for HuggingFace datasets ('train' or 'test')
        """
        self.dataset_path = dataset_path
        self.split = split
        
        # Initialize components
        self.loader = LoaderFactory.create(loader_type)
        
        # Default extractor to loader type if not specified
        if extractor_type is None:
            extractor_type = loader_type
        self.extractor = ExtractorFactory.create(extractor_type)
        
        # Initialize grader
        if grader_type.lower() == 'exact':
            self.grader = ExactMatchGrader()
        elif grader_type.lower() == 'math':
            self.grader = MathGrader()
        else:
            raise ValueError(f"Unknown grader type: {grader_type}. Use 'exact' or 'math'")
        
        # Load problems into a dictionary
        self.problems_dict: Dict[str, MathProblem] = {}
        self._load_problems()
        
        print(f"Initialized MathResponseRegrader:")
        print(f"  Dataset: {dataset_path}")
        print(f"  Loader: {self.loader.get_name()}")
        print(f"  Extractor: {self.extractor.get_name()}")
        print(f"  Grader: {self.grader.get_name()}")
        print(f"  Loaded {len(self.problems_dict)} problems")
    
    def _load_problems(self):
        """Load problems from dataset and create a lookup dictionary"""
        problems = self.loader.load(self.dataset_path, split=self.split)
        
        for problem in problems:
            self.problems_dict[problem.problem_id] = problem
        
        print(f"Created problem lookup dictionary with {len(self.problems_dict)} entries")
    
    def load_metrics_file(self, metrics_path: str) -> List[Dict[str, Any]]:
        """Load metrics from JSONL file"""
        metrics_records = []
        
        # Count lines first for progress bar
        with open(metrics_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        with open(metrics_path, 'r') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Loading metrics"), 1):
                try:
                    record = json.loads(line.strip())
                    metrics_records.append(record)
                except json.JSONDecodeError as e:
                    print(f"\nWarning: Failed to parse line {line_num}: {e}")
        
        print(f"Loaded {len(metrics_records)} metrics records from {metrics_path}")
        return metrics_records
    
    def regrade_single_response(
        self,
        record: Dict[str, Any],
        problem: MathProblem
    ) -> Dict[str, Any]:
        """Re-grade a single response"""
        response_text = record.get('response_text', '')
        
        # Extract answer from response
        extracted_answer = self.extractor.extract(response_text)
        
        # Grade the extracted answer
        is_correct = self.grader.grade(extracted_answer, problem.answer)
        
        # Update record with new grading results
        updated_record = record.copy()
        updated_record['extracted_answer'] = extracted_answer
        updated_record['is_correct'] = is_correct
        updated_record['ground_truth'] = problem.answer
        
        return updated_record
    
    def regrade_all_responses(
        self,
        metrics_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-grade all responses"""
        updated_records = []
        skipped = 0
        failed = 0
        
        for record in tqdm(metrics_records, desc="Re-grading"):
            problem_id = record.get('problem_id')
            
            # Find the corresponding problem
            if problem_id not in self.problems_dict:
                print(f"Warning: Problem {problem_id} not found in dataset, skipping...")
                updated_records.append(record)  # Keep original
                skipped += 1
                continue
            
            problem = self.problems_dict[problem_id]
            
            try:
                updated_record = self.regrade_single_response(record, problem)
                updated_records.append(updated_record)
            except Exception as e:
                print(f"\nError re-grading record for problem {problem_id}: {e}")
                updated_records.append(record)  # Keep original
                failed += 1
        
        if skipped > 0:
            print(f"Skipped {skipped} records due to missing problems")
        if failed > 0:
            print(f"Failed to re-grade {failed} records, kept original values")
        
        return updated_records
    
    def save_metrics_file(
        self,
        metrics_records: List[Dict[str, Any]],
        output_path: str
    ):
        """Save updated metrics to JSONL file"""
        with open(output_path, 'w') as f:
            for record in tqdm(metrics_records, desc="Saving metrics"):
                f.write(json.dumps(record) + '\n')
        
        print(f"Saved {len(metrics_records)} updated records to {output_path}")
    
    def regrade_metrics_file(
        self,
        metrics_path: str,
        output_path: Optional[str] = None,
        backup: bool = True
    ):
        """
        Re-grade all responses in a metrics file
        
        Args:
            metrics_path: Path to the metrics JSONL file
            output_path: Path to save updated metrics (defaults to overwriting original)
            backup: Whether to create a backup of the original file
        """
        if output_path is None:
            output_path = metrics_path
        
        # Create backup if requested and overwriting
        if backup and output_path == metrics_path:
            backup_path = metrics_path + '.backup'
            if os.path.exists(backup_path):
                print(f"Warning: Backup file {backup_path} already exists, skipping backup")
            else:
                import shutil
                shutil.copy2(metrics_path, backup_path)
                print(f"Created backup: {backup_path}")
        
        # Load metrics
        metrics_records = self.load_metrics_file(metrics_path)
        
        if not metrics_records:
            print("No records to re-grade!")
            return
        
        # Re-grade all responses
        updated_records = self.regrade_all_responses(metrics_records)
        
        # Save updated metrics
        self.save_metrics_file(updated_records, output_path)
        
        # Print summary
        total = len(updated_records)
        correct = sum(1 for r in updated_records if r.get('is_correct', False))
        
        print(f"\n{'='*60}")
        print(f"Re-grading complete!")
        print(f"Total responses: {total}")
        print(f"Correct responses: {correct} ({correct/total*100:.1f}%)")
        print(f"Updated file: {output_path}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-grade math responses in metrics JSONL files"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset JSONL file (e.g., datasets/aime.jsonl, datasets/gsm8k_1000.jsonl)"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to the all_response_metrics.jsonl file to re-grade"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save updated metrics (defaults to overwriting input)"
    )
    
    parser.add_argument(
        "--loader",
        type=str,
        default="aime",
        choices=["aime", "math", "gsm8k", "generic"],
        help="Type of dataset loader to use"
    )
    
    parser.add_argument(
        "--extractor",
        type=str,
        default=None,
        choices=["aime", "math", "gsm8k", "generic"],
        help="Type of answer extractor to use (defaults to same as loader)"
    )
    
    parser.add_argument(
        "--grader",
        type=str,
        default="math",
        choices=["exact", "math"],
        help="Type of grader to use (exact match or math equivalence)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use for HuggingFace datasets (e.g., 'train' or 'test')"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup of the original metrics file"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    # Only check if dataset exists if it looks like a file path (not a HuggingFace dataset ID)
    if args.dataset.endswith(('.json', '.jsonl')) or '/' in args.dataset and not args.dataset.count('/') == 1:
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset file not found: {args.dataset}")
            sys.exit(1)
    
    if not os.path.exists(args.metrics):
        print(f"Error: Metrics file not found: {args.metrics}")
        sys.exit(1)
    
    # Initialize regrader
    regrader = MathResponseRegrader(
        dataset_path=args.dataset,
        loader_type=args.loader,
        extractor_type=args.extractor,
        grader_type=args.grader,
        split=args.split,
    )
    
    # Re-grade the metrics file
    regrader.regrade_metrics_file(
        metrics_path=args.metrics,
        output_path=args.output,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
