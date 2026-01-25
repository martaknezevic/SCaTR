"""
Evaluate BigCodeBench responses from regenerate_metrics.py output

This script takes the JSONL output from regenerate_metrics.py and evaluates
each response using the BigCodeBench test suite.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Import local py_executor
from py_executor import PyExecutor


def load_dataset(dataset_path: str) -> Dict[str, Dict]:
    """Load BigCodeBench dataset and create problem ID to problem mapping"""
    problems = {}
    
    with open(dataset_path, 'r') as f:
        for line in f:
            problem = json.loads(line.strip())
            problems[problem['task_id']] = problem
    
    return problems


def load_responses(responses_path: str) -> List[Dict]:
    """Load responses from regenerate_metrics.py output"""
    responses = []
    
    with open(responses_path, 'r') as f:
        for line in f:
            responses.append(json.loads(line.strip()))
    
    return responses


def evaluate_responses(
    responses: List[Dict],
    problems: Dict[str, Dict],
    timeout: int = 20
):
    """Evaluate each response against corresponding problem tests"""
    
    # Initialize executor
    exe = PyExecutor()
    
    results = []
    
    for response in tqdm(responses, desc="Evaluating responses"):
        problem_id = response['problem_id']
    
        
        if problem_id not in problems:
            print(f"Warning: Problem {problem_id} not found in dataset")
            continue
        
        problem = problems[problem_id]
        extracted_code = response.get('extracted_code', '')
        
        # Get test code and imports from problem
        test_code = problem.get('test', '')
        imports = problem.get('imports', None)
        entry_point = problem.get('entry_point', '')
        
        # Evaluate the response code using evaluate_bigcode_bench
        is_passing = exe.evaluate_bigcode_bench(
            name=entry_point,
            func=extracted_code,
            tests=test_code,
            timeout=timeout,
            imports=imports
        )
        
        # Add evaluation results to response
        result = response.copy()
        result['is_correct'] = is_passing
        
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BigCodeBench responses from regenerate_metrics.py output"
    )
    
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to responses JSONL file (output from regenerate_metrics.py)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to BigCodeBench dataset JSONL file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file with evaluation results"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout for code execution (default: 10 seconds)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset}...")
    problems = load_dataset(args.dataset)
    print(f"Loaded {len(problems)} problems")
    
    print(f"Loading responses from {args.responses}...")
    responses = load_responses(args.responses)
    print(f"Loaded {len(responses)} responses")
    
    print(f"Evaluating responses...")
    results = evaluate_responses(responses, problems, timeout=args.timeout)
    
    print(f"Writing results to {args.output}...")
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print summary statistics
    total = len(results)
    if total > 0:
        passed = sum(1 for r in results if r.get('is_passing', False))
        print(f"\nEvaluation complete!")
        print(f"Total responses: {total}")
        print(f"Passed: {passed} ({100*passed/total:.1f}%)")
        print(f"Failed: {total - passed} ({100*(total-passed)/total:.1f}%)")
        print(f"Results saved to: {args.output}")
    else:
        print(f"\nNo results to evaluate!")


if __name__ == "__main__":
    main()
