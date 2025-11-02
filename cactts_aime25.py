"""
AIME 2025 Evaluation Script with Hugging Face Dataset Support

This script evaluates language models on AIME (American Invitational Mathematics Examination) 
problems using multiple response generation and tail confidence selection strategies.

Installation Requirements:
    pip install openai numpy datasets

Usage Examples:

1. Using Hugging Face dataset:
   python script.py
   Enter dataset source: AI-MO/aimo-validation-aime
   
2. Using local JSON file:
   python script.py  
   Enter dataset source: /path/to/aime_data.json
   
3. Using sample data for testing:
   python script.py
   Enter dataset source: [press Enter]

Popular AIME/Math Datasets on HF:
- AI-MO/aimo-validation-aime
- hendrycks/competition_math  
- lighteval/MATH
- AI-MO/NuminaMath-CoT
"""
import time
import openai
import numpy as np
import asyncio
import json
import random
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import openai
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

# Import modular components
from metrics import ConfidenceMetrics, aggregate_metrics
from storage import ChoiceStorage, MetricsStorage, ResultsAggregator
from openai.types.chat.chat_completion import ChoiceLogprobs

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

@dataclass
class AIMEProblem:
    """Structure for AIME problem"""
    problem_id: str
    problem_text: str
    answer: int
    year: int = 2025

@dataclass
class ResponseResult:
    """Structure for response evaluation"""
    response_text: str
    extracted_answer: int
    is_correct: bool
    total_tokens: int
    choice_index: int
    aggregated_metrics: Dict[str, float]  # All metrics flattened

class AIMEEvaluator:
    def __init__(self, client, model_id: str, output_dir: str, streaming=False, choices_dir: str = 'choices', metrics_file: str = 'all_response_metrics.jsonl', strategies_file: str = 'strategy_results.json'):
        self.client = client
        self.model_id = model_id
        self.output_dir = output_dir
        self.metrics_calculator = ConfidenceMetrics()
        self.streaming = streaming
        
        # Storage paths
        self.choices_dir = os.path.join(output_dir, choices_dir)
        self.metrics_file = os.path.join(output_dir, metrics_file)
        self.strategies_file = os.path.join(output_dir, strategies_file)

    def load_aime_dataset(self, dataset_source: str, dataset_config: Optional[str] = None, split: str = "test") -> List[AIMEProblem]:
        """
        Load AIME 2025 dataset from Hugging Face or local JSON file.
        
        Args:
            dataset_source: HF dataset name (e.g., "AI-MO/aimo-validation-aime") or local file path
            dataset_config: Configuration name for HF dataset (optional)
            split: Dataset split to use (default: "test")
        """
        # return self.create_sample_dataset()
        # Try loading from Hugging Face first
        if HF_AVAILABLE and not dataset_source.endswith('.json'):
            try:
                print(f"Loading dataset from Hugging Face: {dataset_source}")
                if dataset_config:
                    dataset = load_dataset(dataset_source, dataset_config, split=split)
                else:
                    dataset = load_dataset(dataset_source, split=split)
                
                problems = []
                for i, item in enumerate(dataset):
                    # Handle different possible column names
                    problem_text = self._extract_field(item, ['problem', 'question', 'text', 'prompt'])
                    answer = self._extract_field(item, ['answer', 'solution', 'target'])
                    problem_id = self._extract_field(item, ['id', 'problem_id', 'index'], default=f"aime_2025_{i}")
                    
                    if problem_text is None or answer is None:
                        print(f"Warning: Skipping item {i} due to missing problem or answer")
                        continue
                    
                    # Convert answer to integer if it's a string
                    if isinstance(answer, str):
                        # Extract number from answer string
                        answer_match = re.search(r'\d+', str(answer))
                        if answer_match:
                            answer = int(answer_match.group())
                        else:
                            print(f"Warning: Could not extract numeric answer from: {answer}")
                            continue
                    
                    problems.append(AIMEProblem(
                        problem_id=str(problem_id),
                        problem_text=str(problem_text),
                        answer=int(answer)
                    ))
                
                print(f"Successfully loaded {len(problems)} problems from HF dataset")
                return problems
                
            except Exception as e:
                print(f"Failed to load from HF dataset: {e}")
                print("Falling back to local file or sample data...")
        
        # Try loading from local JSON file
        if dataset_source.endswith('.json'):
            try:
                with open(dataset_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                problems = []
                for item in data:
                    problems.append(AIMEProblem(
                        problem_id=item["id"],
                        problem_text=item["problem"],
                        answer=item["answer"]
                    ))
                return problems
            except FileNotFoundError:
                print(f"Dataset file {dataset_source} not found.")
        
        # Create sample dataset if all else fails
        print("Creating sample problems for testing...")
        return self.create_sample_dataset()
    
    def _extract_field(self, item: Dict, possible_keys: List[str], default=None):
        """Extract field from item dict using possible key names"""
        for key in possible_keys:
            if key in item and item[key] is not None:
                return item[key]
        return default
    
    def create_sample_dataset(self) -> List[AIMEProblem]:
        """Create sample AIME problems for testing"""
        sample_problems = [
            AIMEProblem(
                problem_id="aime_2025_sample_1",
                problem_text="In the sentence 'This sentence has five words', how many words are there?",
                answer=5
            ),
            AIMEProblem(
                problem_id="aime_2025_sample_2", 
                problem_text="What is 15 + 28?",
                answer=43
            ),
            AIMEProblem(
                problem_id="aime_2025_sample_3",
                problem_text="Find the value of x if 2x + 7 = 19.",
                answer=6
            )
        ]
        return sample_problems
    
    def extract_answer_from_response(self, response_text: str) -> int:
        """
        Extract numerical answer from response text.
        AIME answers are integers from 0 to 999.
        """
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
        
        # If no clear answer found, return -1 to indicate extraction failure
        return -1
    
    async def generate_single_response_stream(self, config: Dict[str, Any]) -> Choice:
        """Generate a single response with streaming and accumulate all data including logprobs"""
        stream_config = {**config, "stream": True, "stream_options": {"include_usage": True}}
        
        # Ensure logprobs are requested
        if "logprobs" not in stream_config or not stream_config.get("logprobs"):
            stream_config["logprobs"] = True
            stream_config["top_logprobs"] = config.get("top_logprobs", 10)
        
        stream = await self.client.chat.completions.create(**stream_config)
        
        # Accumulators
        accumulated_content = ""
        accumulated_logprobs_content = []
        finish_reason = None
        index = 0
        
        async for chunk in stream:
            if not chunk.choices or len(chunk.choices) == 0:
                continue
                
            choice = chunk.choices[0]
            index = choice.index
            
            # Accumulate content
            if hasattr(choice, 'delta') and choice.delta:
                if hasattr(choice.delta, 'content') and choice.delta.content:
                    accumulated_content += choice.delta.content
            
            # Accumulate logprobs token by token
            if hasattr(choice, 'logprobs') and choice.logprobs:
                if hasattr(choice.logprobs, 'content') and choice.logprobs.content:
                    accumulated_logprobs_content.extend(choice.logprobs.content)
            
            # Capture finish reason
            if hasattr(choice, 'finish_reason') and choice.finish_reason:
                finish_reason = choice.finish_reason
        
        # Create ChatCompletionMessage
        message = ChatCompletionMessage(
            content=accumulated_content,
            role="assistant",
            function_call=None,
            tool_calls=None
        )
        
        # Create ChoiceLogprobs
        
        logprobs = ChoiceLogprobs(
            content=accumulated_logprobs_content if accumulated_logprobs_content else None,
            refusal=None
        )
        
        # Create Choice using OpenAI's class
        final_choice = Choice(
            finish_reason=finish_reason or "stop",
            index=index,
            logprobs=logprobs,
            message=message
        )
        
        return final_choice

    async def generate_multiple_responses_streaming(self, config: Dict[str, Any], n_gen: int, seed: int) -> List[Choice]:
        """Generate multiple responses concurrently with streaming"""       
        tasks = [self.generate_single_response_stream(config) for _ in range(n_gen)]
        choices = await asyncio.gather(*tasks)
        return choices

    async def generate_single_response(self, config: Dict[str, Any]) -> Any:
        """Generate a single response asynchronously"""
        response = await self.client.chat.completions.create(**config)
        return response.choices[0]
    
    async def generate_multiple_responses(self, config: Dict[str, Any], n_gen: int, seed: int) -> List[Any]:
        """Generate multiple responses concurrently"""
        tasks = [self.generate_single_response(config) for _ in range(n_gen)]
        choices = await asyncio.gather(*tasks)
        return choices
    
    def _extract_message(self, choice):
        """Extract full text message from a vLLM/OpenAI-style choice object."""
        if not hasattr(choice, "logprobs") or not hasattr(choice.logprobs, "content"):
            return ""
        return "".join(cct.token for cct in choice.logprobs.content if hasattr(cct, "token"))


    def evaluate_responses(self, problem: AIMEProblem, choices: List[Any], tail_n: int = 512, group_size: int = 1024, k: Optional[int] = None) -> List[ResponseResult]:
        """Evaluate all responses for a problem"""
        results = []
        all_metrics_records = []
                
        for i, choice in enumerate(choices):
            # Compute all metrics
            metric_sequences = self.metrics_calculator.compute_all_metrics(choice, k)
            
            # Aggregate metrics (full, tail, group-based)
            aggregated = aggregate_metrics(
                metric_sequences,
                tail_n=tail_n,
                group_size=group_size
            )
            
            # Extract answer and check correctness
            # response_text = choice.message.content
            response_text = self._extract_message(choice)
            if response_text is not None:
                extracted_answer = self.extract_answer_from_response(response_text)
            else:
                extracted_answer = None
            is_correct = extracted_answer == problem.answer
            
            # Create result
            result = ResponseResult(
                response_text=response_text,
                extracted_answer=extracted_answer,
                is_correct=is_correct,
                total_tokens=len(metric_sequences.get('mean', [])),
                choice_index=i,
                aggregated_metrics=aggregated
            )
            results.append(result)
            
            # Prepare metrics record for storage
            metrics_record = {
                'problem_id': problem.problem_id,
                'rollout_idx': i,
                'response_text': response_text,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'expected_answer': problem.answer,
                'total_tokens': result.total_tokens,
                **aggregated
            }
            all_metrics_records.append(metrics_record)
        
        # Save metrics to JSONL
        MetricsStorage.save_all_metrics_batch(
            all_metrics_records,
            self.metrics_file,
            use_compression=False
        )
        
        return results
    
    def apply_selection_strategies(self, results: List[ResponseResult]) -> Dict[str, ResponseResult]:
        """Apply different selection strategies including exponential confidence"""
        strategies = {}
        
        # Random selection
        strategies['random'] = random.choice(results)
        
        # Oracle selection (best correct answer, or best if none correct)
        correct_results = [r for r in results if r.is_correct]
        if correct_results:
            strategies['oracle'] = max(correct_results, key=lambda x: x.aggregated_metrics['mean_tail'])
        else:
            strategies['oracle'] = max(results, key=lambda x: x.aggregated_metrics['mean_tail'])

        # Generate strategies for all metric aggregations
        metric_keys = list(results[0].aggregated_metrics.keys()) if results else []
        
        for metric_key in metric_keys:
            # Highest strategy
            strategies[f'highest_{metric_key}'] = max(
                results, 
                key=lambda x: x.aggregated_metrics.get(metric_key, 0)
            )
            
            # Lowest strategy
            strategies[f'lowest_{metric_key}'] = min(
                results,
                key=lambda x: x.aggregated_metrics.get(metric_key, 0)
            )        
        return strategies

    async def evaluate_problem(self, problem: AIMEProblem, n_gen: int, tail_n: int, group_size: int, temperature: float, top_logprobs: int, seed: int) -> Dict[str, Any]:
        """Evaluate a single AIME problem"""
        print(f"\nEvaluating Problem {problem.problem_id}")
        print(f"Expected answer: {problem.answer}")
        
        config = {
            "model": self.model_id,
            # "max_tokens": 32768,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are solving an AIME (American Invitational Mathematics Examination) problem. Show your work step by step and provide your final numerical answer clearly. AIME answers are integers from 0 to 999."
                },
                {
                    "role": "user", 
                    "content": problem.problem_text
                }
            ],
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "extra_body": {"reasoning_effort": "high"}
        }
        
        # Generate responses
        print(f"Generating {n_gen} responses...")
        time_start = time.time()
        if self.streaming:
            choices = await self.generate_multiple_responses_streaming(config, n_gen, seed)
        else:
            choices = await self.generate_multiple_responses(config, n_gen, seed)
        time_end = time.time()
        print(f"Generation {problem.problem_id} complete in.............. {time_end - time_start:.1f}s")
        
        time_start = time.time()
        # Save full choices to disk
        ChoiceStorage.save_choices(
            problem.problem_id,
            choices,
            self.choices_dir,
            use_compression=False
        )
        time_end = time.time()
        print(f"Saved choices for {problem.problem_id} in.............. {time_end - time_start:.1f}s")
        
        # Evaluate responses
        
        time_start = time.time()
        results = self.evaluate_responses(problem, choices, tail_n, group_size)
        
        # Apply selection strategies
        strategies = self.apply_selection_strategies(results)
        time_end = time.time()
        print(f"Evaluation {problem.problem_id} complete in.............. {time_end - time_start:.1f}s")
        
        return {
            'problem': problem,
            'results': results,
            'strategies': strategies,
            'n_correct': sum(1 for r in results if r.is_correct),
            'total_generated': len(results)
        }

    async def evaluate_problem_with_semaphore(self, problem: AIMEProblem, n_gen: int, tail_n: int, group_size: int, temperature: float, top_logprobs: int, semaphore: asyncio.Semaphore, seed: int) -> Dict[str, Any]:
        """Evaluate a single problem with semaphore control"""
        async with semaphore:
            return await self.evaluate_problem(problem, n_gen, tail_n, group_size, temperature=temperature, top_logprobs=top_logprobs, seed=seed)

    async def evaluate_problems_batch(self, problems: List[AIMEProblem], n_gen: int, tail_n: int, group_size: int,
                                    max_concurrent: int = 3, temperature: float = 0.6, top_logprobs: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
        """Evaluate a batch of problems concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all problems in the batch
        tasks = [
            self.evaluate_problem_with_semaphore(problem, n_gen, tail_n, group_size, temperature=temperature, top_logprobs=top_logprobs, semaphore=semaphore, seed=seed) 
            for problem in problems
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error evaluating problem {problems[i].problem_id}: {result}")
                # Create a dummy result for failed evaluations
                dummy_result = {
                    'problem': problems[i],
                    'results': [],
                    'strategies': {
                        'random': None,
                        'oracle': None
                    },
                    'n_correct': 0,
                    'total_generated': 0,
                    'error': str(result)
                }
                successful_results.append(dummy_result)
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def evaluate_dataset(self, dataset_source: str, n_gen: int, tail_n: int = 2048, group_size: int = 1024,
                              dataset_config: Optional[str] = None, split: str = "test",
                              max_concurrent_problems: int = 3, batch_size: int = 10, temperature: float = 0.6, top_logprobs: int = 10, seed: int = 42) -> Dict[str, Any]:
        """
        Evaluate entire AIME dataset with concurrent problem processing
        
        Args:
            dataset_source: HF dataset name or local file path
            n_gen: Number of generations per problem
            tail_n: Tail confidence window size
            dataset_config: HF dataset configuration
            split: Dataset split to use
            max_concurrent_problems: Max concurrent problem evaluations
            batch_size: Number of problems to process in each batch
        """
        problems = self.load_aime_dataset(dataset_source, dataset_config, split)
        
        print(f"Loaded {len(problems)} problems for task evaluation")
        print(f"Will generate {n_gen} responses per problem")
        print(f"Using tail-{tail_n} confidence for selection strategies")
        print(f"Processing {max_concurrent_problems} problems concurrently in batches of {batch_size}")
        print(f"📊 Task Success Metric: # of problems solved correctly out of {len(problems)} total")
        #print(f"🎯 Strategies to evaluate: Random, Standard Confidence (High/Low), Exponential Confidence (High/Low), Oracle")
        
        all_results = []
        
        # Process problems in batches
        total_batches = (len(problems) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(problems), batch_size):
            batch_end = min(batch_idx + batch_size, len(problems))
            current_batch = problems[batch_idx:batch_end]
            current_batch_num = batch_idx // batch_size + 1
            
            print(f"\n{'='*60}")
            print(f"Processing Batch {current_batch_num}/{total_batches}")
            print(f"Problems {batch_idx + 1}-{batch_end} of {len(problems)}")
            print(f"{'='*60}")
            
            # Process current batch concurrently
            try:
                batch_results = await self.evaluate_problems_batch(
                    current_batch, n_gen, tail_n, group_size, max_concurrent_problems, temperature=temperature, top_logprobs=top_logprobs, seed=seed
                )
                
                all_results.extend(batch_results)
                
                # Print batch summary
                batch_correct = sum(1 for r in batch_results if r.get('n_correct', 0) > 0)
                print(f"\nBatch {current_batch_num} complete:")
                print(f"  Problems with ≥1 correct: {batch_correct}/{len(current_batch)}")
                
            except Exception as e:
                print(f"Error in batch {current_batch_num}: {e}")
                continue
        
        # Aggregate results
        dataset_info = {
            'source': dataset_source,
            'config': dataset_config,
            'split': split,
            'total_problems': len(problems),
            'n_gen': n_gen,
            'tail_n': tail_n,
            'group_size': group_size
        }
        
        # Save strategy-level results
        strategy_results = ResultsAggregator.aggregate_strategy_results(
            all_results,
            dataset_info,
            self.strategies_file
        )
        
        print(f"\n{'='*80}")
        print("RESULTS SAVED")
        print(f"{'='*80}")
        print(f"Strategy results: {self.strategies_file}")
        print(f"All response metrics: {self.metrics_file}")
        print(f"Full choices: {self.choices_dir}/")
        print(f"{'='*80}\n")
        
        return {
            'problem_results': all_results,
            'strategy_results': strategy_results,
            'dataset_info': dataset_info
        }
        
    def _evaluate_problem_worker(self, problem, choices_filename, tail_n, group_size, topk_logprobs=None) -> Dict[str, Any]:
        """
        Worker function that runs in separate process.
        Must be a top-level function or static method for pickling.
        """
        # Load choices
        choices = ChoiceStorage.load_choices(choices_filename)
        
        # Evaluate responses
        results = self.evaluate_responses(problem, choices, tail_n, group_size, topk_logprobs)
        
        # Apply selection strategies
        strategies = self.apply_selection_strategies(results)
        
        return {
            'problem': problem,
            'results': results,
            'strategies': strategies,
            'n_correct': sum(1 for r in results if r.is_correct),
            'total_generated': len(results)
        }
    
    async def evaluate_dataset_offline(self, dataset_source: str, stored_choices_dir: str, topk_logprobs: Optional[int] = None, tail_n: int = 2048, 
                                   group_size: int = 1024, split: str = 'train', 
                                   max_concurrent_problems: int = 30) -> Dict[str, Any]:
        """
        Evaluate entire AIME dataset based on generated choices already stored on disk.
        All problems are processed in parallel with controlled concurrency.
        
        Args:
            dataset_source: HF dataset name or local file path
            stored_choices_dir: Directory containing stored choices
            tail_n: Tail confidence window size
            group_size: Group size for evaluation
            split: Dataset split to use
            max_concurrent_problems: Maximum number of problems to process concurrently
        """
        
        problems = self.load_aime_dataset(dataset_source, None, split)
        
        print(f"Loaded {len(problems)} problems for task evaluation")
        print(f"Using tail-{tail_n} confidence for selection strategies")
        print(f"Processing up to {max_concurrent_problems} problems concurrently")
        print(f"📊 Task Success Metric: # of problems solved correctly out of {len(problems)} total")
        
        # Use ProcessPoolExecutor for CPU-bound work
        num_workers = min(max_concurrent_problems, os.cpu_count() or 1)
        print(f"Using {num_workers} worker processes")
        
        # Progress tracking
        completed = {'count': 0}
        completed_lock = asyncio.Lock()
        
        # Create process pool
        loop = asyncio.get_event_loop()
        executor = ProcessPoolExecutor(max_workers=num_workers)
        
        try:
            async def evaluate_single_problem_offline(problem, idx):
                """Evaluate a single problem with stored choices"""
                try:
                    print(f"\n[{idx + 1}/{len(problems)}] Starting evaluation for Problem {problem.problem_id}")
                    
                    choices_filename = os.path.join(stored_choices_dir, f"{problem.problem_id}_choices.pkl")
                    
                    # Run entire evaluation in process pool
                    result = await loop.run_in_executor(
                        executor,
                        self._evaluate_problem_worker,
                        problem, choices_filename, tail_n, group_size, topk_logprobs
                    )
                    
                    async with completed_lock:
                        completed['count'] += 1
                        n_correct = result['n_correct']
                        status = "✓ HAS CORRECT" if n_correct > 0 else "✗ NO CORRECT"
                        print(f"[{completed['count']}/{len(problems)}] Completed Problem {problem.problem_id}: {n_correct}/{result['total_generated']} correct - {status}")
                    
                    return result
                    
                except Exception as e:
                    print(f"Error evaluating problem {problem.problem_id}: {e}")
                    async with completed_lock:
                        completed['count'] += 1
                    return {
                        'problem': problem,
                        'error': str(e),
                        'n_correct': 0,
                        'total_generated': 0
                    }
            
            # Create tasks for all problems
            tasks = [
                evaluate_single_problem_offline(problem, idx)
                for idx, problem in enumerate(problems)
            ]
            
            print(f"\n{'='*60}")
            print(f"Starting parallel evaluation of all {len(problems)} problems...")
            print(f"{'='*60}\n")
            
            # Run all tasks concurrently
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            # Clean up executor
            executor.shutdown(wait=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                print(f"Exception in problem {i}: {result}")
                valid_results.append({
                    'problem': problems[i],
                    'error': str(result),
                    'n_correct': 0,
                    'total_generated': 0
                })
            else:
                valid_results.append(result)
        
        # Print final summary
        total_correct = sum(1 for r in valid_results if r.get('n_correct', 0) > 0)
        total_responses = sum(r.get('total_generated', 0) for r in valid_results)
        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total problems: {len(problems)}")
        print(f"Problems with ≥1 correct: {total_correct}/{len(problems)}")
        print(f"Total responses evaluated: {total_responses}")
        print(f"{'='*80}\n")
        
        # Aggregate results
        dataset_info = {
            'source': dataset_source,
            'split': split,
            'total_problems': len(problems),
            'tail_n': tail_n,
            'group_size': group_size,
        }
        
        # Save strategy-level results
        strategy_results = ResultsAggregator.aggregate_strategy_results(
            valid_results,
            dataset_info,
            self.strategies_file
        )
        
        print(f"\n{'='*80}")
        print("RESULTS SAVED")
        print(f"{'='*80}")
        print(f"Strategy results: {self.strategies_file}")
        print(f"All response metrics: {self.metrics_file}")
        print(f"Full choices: {self.choices_dir}/")
        print(f"{'='*80}\n")
        
        return {
            'problem_results': valid_results,
            'strategy_results': strategy_results,
            'dataset_info': dataset_info
        }

def parse_args():
    parser = argparse.ArgumentParser(description="AIME Evaluator Configuration")
    
    # Dataset
    parser.add_argument("--dataset_source", type=str, default="MathArena/aime_2025")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    
    # Generation
    parser.add_argument("--n_gen", type=int, default=5)
    parser.add_argument("--tail_n", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_logprobs", type=int, default=10)
    parser.add_argument("--streaming", action='store_true', help="Enable streaming generation")
    
    # Performance
    parser.add_argument("--max_concurrent", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=10)
    
    # Output
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

async def main():
    print("AIME 2025 Evaluation Script")
    print("="*50)
    
    if not HF_AVAILABLE:
        print("⚠️  Hugging Face datasets not available. Install with: pip install datasets")
        print("Falling back to local JSON file support only.\n")
    
    args = parse_args()
    
    # Setup output directory
    dataset_name = args.dataset_source.replace('/', '_').replace('.json', '')
    output_dir = os.path.join(args.output_dir, dataset_name, f'turn{args.turn}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Initialize client
    client = openai.AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="token-abc",
        timeout=30 * 60 * 2,
    )
    
    # Get model
    models = await client.models.list()
    model_id = models.data[0].id
    print(f"Using model: {model_id}")
    
    # Initialize evaluator
    evaluator = AIMEEvaluator(client, model_id, output_dir, streaming=args.streaming)
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING AIME 2025 TASK PERFORMANCE EVALUATION")
    print("Goal: Measure how many problems each strategy solves correctly")
    print("="*80)
    
    import time
    start_time = time.time()
    
    results = await evaluator.evaluate_dataset(
        dataset_source=args.dataset_source,
        n_gen=args.n_gen,
        tail_n=args.tail_n,
        group_size=args.group_size,
        dataset_config=args.dataset_config,
        split=args.split,
        max_concurrent_problems=args.max_concurrent,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total time: {execution_time:.1f}s ({execution_time/60:.1f} min)")
    print(f"Problems evaluated: {results['dataset_info']['total_problems']}")
    
    strategy_results = results['strategy_results']['strategy_results']
    if strategy_results:
        best = max(strategy_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest strategy: {best[0]}")
        print(f"  Accuracy: {best[1]['accuracy']:.2%}")
        print(f"  Correct: {best[1]['correct']}/{best[1]['total']}")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())