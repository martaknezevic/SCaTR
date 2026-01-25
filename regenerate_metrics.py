"""
Regenerate metrics from existing choices directory
"""

import os
import pickle
import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from code_evaluator import (
    CodeEvaluationConfig,
    ExtractorFactory,
    GraderFactory,
    CodeProblem,
)
from metrics import ConfidenceMetrics, aggregate_metrics
from storage import MetricsStorage


async def process_choice_file(choice_file: Path, problems_dict: dict, 
                              extractor, grader, metrics_calculator, config):
    """Process a single choice file and generate metrics"""
    
    # Extract problem ID from filename
    problem_id = choice_file.stem.replace('_choices', '')
    
    # Find matching problem
    if problem_id not in problems_dict:
        print(f"Warning: Problem {problem_id} not found in dataset")
        return []
    
    problem = problems_dict[problem_id]
    
    # Load choices
    with open(choice_file, 'rb') as f:
        choices = pickle.load(f)
    
    if not isinstance(choices, list):
        choices = [choices]
    
    print(f"Processing {problem_id}: {len(choices)} choices")
    
    # Step 1: Extract code
    print(f"  Step 1/4: Extracting code from responses...")
    time_start = time.time()
    
    grading_tasks = []
    extracted_codes = []
    response_texts = []
    
    for choice in choices:
        response_text = choice.message.content if choice.message.content else ""
        response_texts.append(response_text)
        extracted_code = extractor.extract(response_text, problem.entry_point)
        extracted_codes.append(extracted_code)
        
        # Create grading task
        task = grader.grade(
            extracted_code if extracted_code else "",
            problem.test_cases,
            config.execution_timeout
        )
        grading_tasks.append(task)
    
    time_extract = time.time() - time_start
    print(f"    ✓ Extraction complete in {time_extract:.2f}s")
    
    # Step 2: Grade code
    print(f"  Step 2/4: Grading {len(grading_tasks)} code submissions...")
    time_start = time.time()
    
    grading_results = await asyncio.gather(*grading_tasks)
    
    time_grading = time.time() - time_start
    print(f"    ✓ Grading complete in {time_grading:.2f}s")
    
    # Step 3: Compute metrics (parallelized)
    print(f"  Step 3/4: Computing confidence metrics in parallel...")
    time_start = time.time()
    
    # Compute metrics for all choices in parallel using thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=min(len(choices), 16)) as executor:
        metric_tasks = [
            loop.run_in_executor(executor, metrics_calculator.compute_all_metrics, choice)
            for choice in choices
        ]
        all_metric_sequences = await asyncio.gather(*metric_tasks)
    
    time_metrics = time.time() - time_start
    print(f"    ✓ Metrics computed in {time_metrics:.2f}s")
    
    # Step 4: Assemble results
    print(f"  Step 4/4: Assembling results...")
    time_start = time.time()
    
    all_metrics_records = []
    
    for choice_idx, (response_text, extracted_code, (is_passing, feedback, partial_tests), metric_sequences) in enumerate(
        zip(response_texts, extracted_codes, grading_results, all_metric_sequences)
    ):
        # Calculate code reward
        code_rew = np.mean(np.asarray(partial_tests, dtype=np.float32))
        
        num_passed = int(np.sum(np.asarray(partial_tests, dtype=np.float32)))
        total_tests = len(partial_tests)
        
        # Aggregate metrics
        aggregated = aggregate_metrics(
            metric_sequences,
            tail_n=config.tail_n,
            group_size=config.group_size
        )
        
        # Count tokens
        total_tokens = len(metric_sequences.get('mean', []))
        
        # Prepare metrics record
        metrics_record = {
            'problem_id': problem.problem_id,
            'rollout_idx': choice_idx,
            'response_text': response_text,
            'extracted_code': extracted_code,
            'is_correct': is_passing,
            'num_tests_passed': num_passed,
            'total_tests': total_tests,
            'total_tokens': total_tokens,
            **aggregated
        }
        all_metrics_records.append(metrics_record)
    
    time_assemble = time.time() - time_start
    print(f"    ✓ Results assembled in {time_assemble:.2f}s")
    
    total_time = time_extract + time_grading + time_metrics + time_assemble
    print(f"  Complete - {len(all_metrics_records)} records in {total_time:.2f}s total")
    
    return all_metrics_records


async def regenerate_metrics(choices_dir: str, dataset_file: str, config_file: str, output_file: str):
    """Regenerate metrics for all choices in directory"""
    
    print("="*80)
    print("REGENERATING METRICS FROM CHOICES")
    print("="*80)
    
    # Load configuration
    print(f"\nLoading configuration from {config_file}")
    config = CodeEvaluationConfig.from_yaml(config_file)
    
    # Initialize components
    grader = GraderFactory.create(config.grader_type, **config.grader_params)
    
    if config.extractor_type.lower() == 'hybrid':
        extractor = ExtractorFactory.create(
            config.extractor_type,
            grader=grader,
            execution_timeout=config.execution_timeout
        )
    else:
        extractor = ExtractorFactory.create(config.extractor_type)
    
    metrics_calculator = ConfidenceMetrics()
    
    print(f"Extractor: {extractor.get_name()}")
    print(f"Grader: {grader.get_name()}")
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_file}")
    problems_dict = {}
    
    if dataset_file.endswith('.jsonl'):
        with open(dataset_file, 'r') as f:
            for line in f:
                problem_data = json.loads(line)
                problem_id = problem_data.get('problem_id', problem_data.get('task_id'))
                
                problem = CodeProblem(
                    problem_id=problem_id,
                    problem_text=problem_data.get('prompt', problem_data.get('problem_text', '')),
                    test_cases=problem_data.get('test', problem_data.get('test_cases', problem_data.get('tests', []))),
                    entry_point=problem_data.get('entry_point'),
                    metadata=problem_data.get('metadata', {})
                )
                problems_dict[problem_id] = problem
    else:
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        for problem_data in data:
            problem_id = problem_data.get('problem_id', problem_data.get('task_id'))
            
            problem = CodeProblem(
                problem_id=problem_id,
                problem_text=problem_data.get('prompt', problem_data.get('problem_text', '')),
                test_cases=problem_data.get('test', problem_data.get('test_cases', problem_data.get('tests', []))),
                entry_point=problem_data.get('entry_point'),
                metadata=problem_data.get('metadata', {})
            )
            problems_dict[problem_id] = problem
    
    print(f"Loaded {len(problems_dict)} problems")
    
    # Get all choice files
    choices_path = Path(choices_dir)
    choice_files = sorted(choices_path.glob("*.pkl"))
    
    if not choice_files:
        print(f"No choice files found in {choices_dir}")
        return
    
    print(f"Found {len(choice_files)} choice files")
    print(f"Writing results incrementally to {output_file}")
    
    # Open output file for writing (create or truncate)
    with open(output_file, 'w') as f:
        pass  # Clear file if exists
    
    # Process all choice files and write incrementally
    total_records = 0
    
    for i, choice_file in enumerate(choice_files, 1):
        print(f"\n[{i}/{len(choice_files)}] Processing {choice_file.name}")
        
        try:
            metrics = await process_choice_file(
                choice_file, problems_dict, extractor, grader, 
                metrics_calculator, config
            )
            print(f"  ✓ Computed metrics for {len(metrics)} records")
            # Write metrics immediately after processing each file
            if metrics:
                print('Writing metrics to output file...')
                MetricsStorage.save_all_metrics_batch(
                    metrics,
                    output_file,
                    use_compression=False
                )
                total_records += len(metrics)
                print(f"  ✓ Wrote {len(metrics)} records (total: {total_records})")
                
        except Exception as e:
            print(f"Error processing {choice_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✓ All metrics saved successfully!")
    print(f"  Total records: {total_records}")
    print(f"  Output file: {output_file}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Regenerate metrics from choices directory")
    
    parser.add_argument("--choices_dir", type=str, required=True,
                       help="Path to choices directory")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset file (.json or .jsonl)")
    parser.add_argument("--config", type=str, default="config_kodcode.yaml",
                       help="Path to config file")
    parser.add_argument("--output", type=str, default="x.jsonl",
                       help="Output metrics file")
    
    args = parser.parse_args()
    
    asyncio.run(regenerate_metrics(
        args.choices_dir,
        args.dataset,
        args.config,
        args.output
    ))


if __name__ == "__main__":
    main()
