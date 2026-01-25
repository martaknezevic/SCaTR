"""
Re-grading Script for Code Responses

This script re-grades all responses in all_response_metrics.jsonl files using
the ExecutorAPIGrader. It loads problems from the original dataset, maps them
to responses by problem_id, extracts code, and re-grades each response.

Usage:
    python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics path/to/all_response_metrics.jsonl --loader humaneval
    python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics path/to/all_response_metrics.jsonl --loader kodcode
"""

import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

# Import components from code_evaluator
from code_evaluator import (
    HumanEvalLoader,
    KodcodeLoader,
    ExecutorAPIGrader,
    LoaderFactory,
    CodeProblem,
    CodeExtractor,
)

# Import code utilities
from code_utils import parse_between_output_tags, extract_function_with_dependencies
import re


class OutputTagExtractor(CodeExtractor):
    """Extractor for code within <output>...</output> tags"""
    
    def get_name(self) -> str:
        return "OutputTag"
    
    def extract(self, response_text: str, entry_point: Optional[str] = None) -> Optional[str]:
        """Extract code from <output> tags"""
        if not response_text:
            return None
        
        # First extract from output tags
        code = parse_between_output_tags(response_text)
        return code


class MarkdownCodeExtractor(CodeExtractor):
    """Extractor for code within ```python``` markdown blocks, extracts only function code"""
    
    def get_name(self) -> str:
        return "MarkdownCode"
    
    def extract(self, response_text: str, entry_point: Optional[str] = None) -> Optional[str]:
        """Extract code from markdown blocks and extract only the function code"""
        if not response_text:
            return None
        
        # Find all ```python or ``` code blocks
        pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if not matches:
            return None
        
        # Use the last code block
        code = matches[-1].strip()
        
        # If entry_point is specified, extract only that function and its dependencies
        if entry_point:
            try:
                extracted = extract_function_with_dependencies(code, entry_point)
                if extracted:
                    return extracted
            except Exception:
                # If extraction fails, fall back to original code
                pass
        
        # If no entry_point or extraction failed, return the code as-is
        # (this maintains backward compatibility for cases without entry_point)
        return code if code else None


class HybridExtractor(CodeExtractor):
    """Hybrid extractor that tries both <output> tags and markdown, picks best result"""
    
    def __init__(self, grader: ExecutorAPIGrader, execution_timeout: int = 10):
        self.output_extractor = OutputTagExtractor()
        self.markdown_extractor = MarkdownCodeExtractor()
        self.grader = grader
        self.execution_timeout = execution_timeout
    
    def get_name(self) -> str:
        return "Hybrid"
    
    async def extract_and_grade(self, response_text: str, entry_point: Optional[str], 
                                test_cases, extractor) -> tuple:
        """Extract code and grade it"""
        code = extractor.extract(response_text, entry_point)
        if not code:
            return None, 0, []
        
        is_passing, feedback, partial_tests = await self.grader.grade(
            code, test_cases, self.execution_timeout
        )
        
        import numpy as np
        num_passed = int(np.sum(np.asarray(partial_tests, dtype=np.float32)))
        
        return code, num_passed, partial_tests
    
    async def extract_with_test(self, response_text: str, entry_point: Optional[str] = None,
                               test_cases = None) -> Optional[str]:
        """Extract using both methods and pick the one that passes more tests"""
        if not response_text or test_cases is None:
            # Fallback to output tag if no test cases provided
            return self.output_extractor.extract(response_text, entry_point)
        
        # Try both extractors
        output_code, output_passed, _ = await self.extract_and_grade(
            response_text, entry_point, test_cases, self.output_extractor
        )
        
        markdown_code, markdown_passed, _ = await self.extract_and_grade(
            response_text, entry_point, test_cases, self.markdown_extractor
        )
        
        # Pick the one with more tests passed
        if output_code is None and markdown_code is None:
            return None
        elif output_code is None:
            return markdown_code
        elif markdown_code is None:
            return output_code
        elif markdown_passed > output_passed:
            return markdown_code
        else:
            return output_code
    
    def extract(self, response_text: str, entry_point: Optional[str] = None) -> Optional[str]:
        """Synchronous extract - just try output tags first, then markdown"""
        # This is for non-async usage
        code = self.markdown_extractor.extract(response_text, entry_point)
        if code:
            return code
        return self.output_extractor.extract(response_text, entry_point)


class ResponseRegrader:
    """Re-grades responses in metrics JSONL files"""
    
    def __init__(
        self,
        dataset_path: str,
        loader_type: str = "humaneval",
        executor_url: str = "http://localhost:8000/execute",
        max_concurrent: int = 6,
        execution_timeout: int = 10,
    ):
        """
        Initialize the regrader
        
        Args:
            dataset_path: Path to the original dataset (JSONL file with problems)
            loader_type: Type of loader ('humaneval', 'kodcode', or 'generic')
            executor_url: URL of the code executor API
            max_concurrent: Maximum concurrent grading tasks
            execution_timeout: Timeout for code execution in seconds
        """
        self.dataset_path = dataset_path
        self.executor_url = executor_url
        self.max_concurrent = max_concurrent
        self.execution_timeout = execution_timeout
        
        # Initialize components
        self.loader = LoaderFactory.create(loader_type)
        self.grader = ExecutorAPIGrader(
            executor_url=executor_url,
            max_concurrent=max_concurrent
        )
        self.extractor = HybridExtractor(
            grader=self.grader,
            execution_timeout=execution_timeout
        )  # Use HybridExtractor
        
        # Load problems into a dictionary
        self.problems_dict: Dict[str, CodeProblem] = {}
        self._load_problems()
        
        print(f"Initialized ResponseRegrader:")
        print(f"  Dataset: {dataset_path}")
        print(f"  Loader: {self.loader.get_name()}")
        print(f"  Extractor: {self.extractor.get_name()}")
        print(f"  Grader: {self.grader.get_name()}")
        print(f"  Loaded {len(self.problems_dict)} problems")
    
    def _load_problems(self):
        """Load problems from dataset and create a lookup dictionary"""
        problems = self.loader.load(self.dataset_path)
        
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
    
    async def regrade_single_response(
        self,
        record: Dict[str, Any],
        problem: CodeProblem
    ) -> Dict[str, Any]:
        """Re-grade a single response"""
        response_text = record.get('response_text', '')
        
        # Extract code from response using hybrid extractor (tests both methods)
        extracted_code = await self.extractor.extract_with_test(
            response_text, 
            problem.entry_point,
            problem.test_cases
        )
        
        # Grade the extracted code
        is_passing, feedback, partial_tests = await self.grader.grade(
            extracted_code if extracted_code else "",
            problem.test_cases,
            self.execution_timeout
        )
        
        # Calculate metrics
        import numpy as np
        num_tests_passed = int(np.sum(np.asarray(partial_tests, dtype=np.float32)))
        total_tests = len(partial_tests)
        
        # Update record with new grading results
        updated_record = record.copy()
        updated_record['extracted_code'] = extracted_code
        updated_record['is_correct'] = is_passing
        updated_record['num_tests_passed'] = num_tests_passed
        updated_record['total_tests'] = total_tests
        
        return updated_record
    
    async def regrade_all_responses(
        self,
        metrics_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-grade all responses in parallel"""
        tasks = []
        skipped = 0
        record_indices = []  # Track which records we're processing
        
        for idx, record in enumerate(metrics_records):
            problem_id = record.get('problem_id')
            
            # Find the corresponding problem
            if problem_id not in self.problems_dict:
                print(f"Warning: Problem {problem_id} not found in dataset, skipping...")
                skipped += 1
                continue
            
            problem = self.problems_dict[problem_id]
            task = self.regrade_single_response(record, problem)
            tasks.append(task)
            record_indices.append(idx)
        
        if skipped > 0:
            print(f"Skipped {skipped} records due to missing problems")
        
        print(f"Re-grading {len(tasks)} responses...")
        
        # Process tasks with progress bar
        successful_records = [None] * len(metrics_records)
        failed = 0
        
        for i, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Re-grading")):
            try:
                result = await task
                if isinstance(result, Exception):
                    raise result
                successful_records[record_indices[i]] = result
            except Exception as e:
                print(f"\nError re-grading record {record_indices[i]}: {e}")
                # Keep original record
                successful_records[record_indices[i]] = metrics_records[record_indices[i]]
                failed += 1
        
        # Fill in skipped records with original data
        for idx, record in enumerate(metrics_records):
            if successful_records[idx] is None:
                successful_records[idx] = record
        
        if failed > 0:
            print(f"Failed to re-grade {failed} records, kept original values")
        
        return successful_records
    
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
    
    async def regrade_metrics_file(
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
        updated_records = await self.regrade_all_responses(metrics_records)
        
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


async def main():
    parser = argparse.ArgumentParser(
        description="Re-grade code responses in metrics JSONL files"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset JSONL file (e.g., datasets/humaneval.jsonl)"
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
        default="humaneval",
        choices=["humaneval", "kodcode", "generic"],
        help="Type of dataset loader to use"
    )
    
    parser.add_argument(
        "--executor-url",
        type=str,
        default="http://localhost:8001/execute",
        help="URL of the code executor API"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=6,
        help="Maximum concurrent grading tasks"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Execution timeout in seconds"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup of the original metrics file"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    if not os.path.exists(args.metrics):
        print(f"Error: Metrics file not found: {args.metrics}")
        sys.exit(1)
    
    # Initialize regrader
    regrader = ResponseRegrader(
        dataset_path=args.dataset,
        loader_type=args.loader,
        executor_url=args.executor_url,
        max_concurrent=args.max_concurrent,
        execution_timeout=args.timeout,
    )
    
    # Re-grade the metrics file
    await regrader.regrade_metrics_file(
        metrics_path=args.metrics,
        output_path=args.output,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    asyncio.run(main())
