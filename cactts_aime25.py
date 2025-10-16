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

import openai
import numpy as np
import asyncio
import json
import random
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

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
    tail_confidence: float
    mean_confidence: float
    exp_tail_confidence: float  # New exponential confidence measure
    exp_mean_confidence: float  # New exponential confidence measure
    total_tokens: int
    choice_index: int

class AIMEEvaluator:
    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = model_id
        
    def load_aime_dataset(self, dataset_source: str, dataset_config: Optional[str] = None, split: str = "test") -> List[AIMEProblem]:
        """
        Load AIME 2025 dataset from Hugging Face or local JSON file.
        
        Args:
            dataset_source: HF dataset name (e.g., "AI-MO/aimo-validation-aime") or local file path
            dataset_config: Configuration name for HF dataset (optional)
            split: Dataset split to use (default: "test")
        """
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
    
    async def generate_single_response(self, config: Dict[str, Any]) -> Any:
        """Generate a single response asynchronously"""
        response = await self.client.chat.completions.create(**config)
        return response.choices[0]
    
    async def generate_multiple_responses(self, config: Dict[str, Any], n_gen: int) -> List[Any]:
        """Generate multiple responses concurrently"""
        tasks = [self.generate_single_response(config) for _ in range(n_gen)]
        choices = await asyncio.gather(*tasks)
        return choices
    
    def calculate_tail_confidence(self, choice: Any, tail_n: int = 512) -> Dict[str, Any]:
        """Calculate tail confidence for a single trace with both standard and exponential measures"""
        if not choice.logprobs or not choice.logprobs.content:
            return {
                'response': choice.message.content,
                'tail_confidence': 0.0,
                'mean_confidence': 0.0,
                'exp_tail_confidence': 0.0,
                'exp_mean_confidence': 0.0,
                'tokens': [],
                'confidences': [],
                'exp_confidences': [],
                'total_tokens': 0
            }
        
        tokens = []
        confidences = []
        exp_confidences = []  # New exponential confidences
        
        # Convert logprobs to confidence
        for token_data in choice.logprobs.content:
            tokens.append(token_data.token)
            
            # Standard confidence calculation (existing method)
            if token_data.top_logprobs:
                # Mean confidence = negative average of top logprobs
                mean_conf = -sum(logprob_data.logprob for logprob_data in token_data.top_logprobs) / len(token_data.top_logprobs)
                confidences.append(mean_conf)
            else:
                # Fallback if no top_logprobs available
                confidences.append(-token_data.logprob)
            
            # Exponential confidence calculation (new method)
            exp_conf = np.exp(token_data.logprob)
            exp_confidences.append(exp_conf)
        
        # Calculate tail confidence (average of bottom N tokens) for both measures
        if len(confidences) >= tail_n:
            tail_confidence = np.mean(confidences[-tail_n:])
            exp_tail_confidence = np.mean(exp_confidences[-tail_n:])
        else:
            tail_confidence = np.mean(confidences) if confidences else 0.0
            exp_tail_confidence = np.mean(exp_confidences) if exp_confidences else 0.0
        
        return {
            'response': choice.message.content,
            'tail_confidence': tail_confidence,
            'mean_confidence': np.mean(confidences) if confidences else 0.0,
            'exp_tail_confidence': exp_tail_confidence,
            'exp_mean_confidence': np.mean(exp_confidences) if exp_confidences else 0.0,
            'tokens': tokens,
            'confidences': confidences,
            'exp_confidences': exp_confidences,
            'total_tokens': len(tokens)
        }
    
    def evaluate_responses(self, problem: AIMEProblem, choices: List[Any], tail_n: int = 512) -> List[ResponseResult]:
        """Evaluate all responses for a problem"""
        results = []
        
        for i, choice in enumerate(choices):
            confidence_data = self.calculate_tail_confidence(choice, tail_n)
            extracted_answer = self.extract_answer_from_response(confidence_data['response'])
            is_correct = extracted_answer == problem.answer
            
            result = ResponseResult(
                response_text=confidence_data['response'],
                extracted_answer=extracted_answer,
                is_correct=is_correct,
                tail_confidence=confidence_data['tail_confidence'],
                mean_confidence=confidence_data['mean_confidence'],
                exp_tail_confidence=confidence_data['exp_tail_confidence'],
                exp_mean_confidence=confidence_data['exp_mean_confidence'],
                total_tokens=confidence_data['total_tokens'],
                choice_index=i
            )
            results.append(result)
        
        return results
    
    def apply_selection_strategies(self, results: List[ResponseResult]) -> Dict[str, ResponseResult]:
        """Apply different selection strategies including exponential confidence"""
        strategies = {}
        
        # Random selection
        strategies['random'] = random.choice(results)
        
        # Standard confidence strategies (existing)
        strategies['highest_confidence'] = max(results, key=lambda x: x.tail_confidence)
        strategies['lowest_confidence'] = min(results, key=lambda x: x.tail_confidence)
        
        # Exponential confidence strategies (new)
        strategies['highest_exp_confidence'] = max(results, key=lambda x: x.exp_tail_confidence)
        strategies['lowest_exp_confidence'] = min(results, key=lambda x: x.exp_tail_confidence)
        
        # Oracle selection (best correct answer, or best if none correct)
        correct_results = [r for r in results if r.is_correct]
        if correct_results:
            strategies['oracle'] = max(correct_results, key=lambda x: x.tail_confidence)
        else:
            strategies['oracle'] = max(results, key=lambda x: x.tail_confidence)
        
        return strategies
    
    async def evaluate_problem(self, problem: AIMEProblem, n_gen: int, tail_n: int) -> Dict[str, Any]:
        """Evaluate a single AIME problem"""
        print(f"\nEvaluating Problem {problem.problem_id}")
        print(f"Expected answer: {problem.answer}")
        
        config = {
            "model": self.model_id,
            # "max_tokens": 8192,
            "temperature": 0.6,
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
            "top_logprobs": 10,
        }
        
        # Generate responses
        print(f"Generating {n_gen} responses...")
        choices = await self.generate_multiple_responses(config, n_gen)
        
        # Evaluate responses
        results = self.evaluate_responses(problem, choices, tail_n)
        
        # Apply selection strategies
        strategies = self.apply_selection_strategies(results)
        
        return {
            'problem': problem,
            'results': results,
            'strategies': strategies,
            'n_correct': sum(1 for r in results if r.is_correct),
            'total_generated': len(results)
        }

    async def evaluate_problem_with_semaphore(self, problem: AIMEProblem, n_gen: int, tail_n: int, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Evaluate a single problem with semaphore control"""
        async with semaphore:
            return await self.evaluate_problem(problem, n_gen, tail_n)
    
    async def evaluate_problems_batch(self, problems: List[AIMEProblem], n_gen: int, tail_n: int, 
                                    max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Evaluate a batch of problems concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all problems in the batch
        tasks = [
            self.evaluate_problem_with_semaphore(problem, n_gen, tail_n, semaphore) 
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
                        'random': ResponseResult("Error", -1, False, 0.0, 0.0, 0.0, 0.0, 0, -1),
                        'highest_confidence': ResponseResult("Error", -1, False, 0.0, 0.0, 0.0, 0.0, 0, -1),
                        'lowest_confidence': ResponseResult("Error", -1, False, 0.0, 0.0, 0.0, 0.0, 0, -1),
                        'highest_exp_confidence': ResponseResult("Error", -1, False, 0.0, 0.0, 0.0, 0.0, 0, -1),
                        'lowest_exp_confidence': ResponseResult("Error", -1, False, 0.0, 0.0, 0.0, 0.0, 0, -1),
                        'oracle': ResponseResult("Error", -1, False, 0.0, 0.0, 0.0, 0.0, 0, -1)
                    },
                    'n_correct': 0,
                    'total_generated': 0,
                    'error': str(result)
                }
                successful_results.append(dummy_result)
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def evaluate_dataset(self, dataset_source: str, n_gen: int, tail_n: int = 2048, 
                              dataset_config: Optional[str] = None, split: str = "test",
                              max_concurrent_problems: int = 3, batch_size: int = 10) -> Dict[str, Any]:
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
        print(f"🎯 Strategies to evaluate: Random, Standard Confidence (High/Low), Exponential Confidence (High/Low), Oracle")
        
        all_results = []
        strategy_accuracy = {
            'random': {'correct': 0, 'total': 0},
            'highest_confidence': {'correct': 0, 'total': 0},
            'lowest_confidence': {'correct': 0, 'total': 0},
            'highest_exp_confidence': {'correct': 0, 'total': 0},  # New strategy
            'lowest_exp_confidence': {'correct': 0, 'total': 0},   # New strategy
            'oracle': {'correct': 0, 'total': 0}
        }
        
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
                    current_batch, n_gen, tail_n, max_concurrent_problems
                )
                
                # Process results from this batch
                for problem_result in batch_results:
                    all_results.append(problem_result)
                    
                    # Skip if this was an error case
                    if 'error' in problem_result:
                        print(f"Skipped problem {problem_result['problem'].problem_id} due to error")
                        continue
                    
                    # Update strategy accuracies
                    for strategy_name, selected_result in problem_result['strategies'].items():
                        strategy_accuracy[strategy_name]['total'] += 1
                        if selected_result.is_correct:
                            strategy_accuracy[strategy_name]['correct'] += 1
                    
                    # Print problem summary with enhanced confidence reporting
                    problem = problem_result['problem']
                    print(f"\n✅ Problem {problem.problem_id} Results:")
                    print(f"  Expected answer: {problem.answer}")
                    print(f"  Correct generations: {problem_result['n_correct']}/{problem_result['total_generated']}")
                    
                    # Show strategy selections with better formatting
                    correct_strategies = []
                    incorrect_strategies = []
                    
                    for strategy_name, selected_result in problem_result['strategies'].items():
                        status = "✓" if selected_result.is_correct else "✗"
                        
                        # Choose appropriate confidence measure for display
                        if 'exp_confidence' in strategy_name:
                            conf_str = f"{selected_result.exp_tail_confidence:.4f}" if selected_result.exp_tail_confidence > 0 else "N/A"
                        else:
                            conf_str = f"{selected_result.tail_confidence:.4f}" if selected_result.tail_confidence > 0 else "N/A"
                            
                        strategy_info = f"{strategy_name} (ans: {selected_result.extracted_answer}, conf: {conf_str})"
                        
                        if selected_result.is_correct:
                            correct_strategies.append(f"    ✓ {strategy_info}")
                        else:
                            incorrect_strategies.append(f"    ✗ {strategy_info}")
                    
                    # Display correct strategies first
                    if correct_strategies:
                        print("  Strategies that got it RIGHT:")
                        for strategy in correct_strategies:
                            print(strategy)
                    
                    if incorrect_strategies:
                        print("  Strategies that got it WRONG:")
                        for strategy in incorrect_strategies:
                            print(strategy)
                
                # Print batch summary with task-focused metrics
                batch_problems_solvable = sum(1 for r in batch_results if r.get('n_correct', 0) > 0)
                
                # Calculate how many problems each strategy solved in this batch
                batch_strategy_performance = {}
                for result in batch_results:
                    if 'error' not in result:
                        for strategy_name, selected_result in result['strategies'].items():
                            if strategy_name not in batch_strategy_performance:
                                batch_strategy_performance[strategy_name] = 0
                            if selected_result.is_correct:
                                batch_strategy_performance[strategy_name] += 1
                
                print(f"\n📊 Batch {current_batch_num} Task Performance Summary:")
                print(f"  Problems with ≥1 correct generation: {batch_problems_solvable}/{len(current_batch)}")
                
                if batch_strategy_performance:
                    print("  Strategy performance in this batch:")
                    for strategy_name, problems_solved in sorted(batch_strategy_performance.items(), key=lambda x: x[1], reverse=True):
                        print(f"    {strategy_name.replace('_', ' ').title()}: {problems_solved}/{len(current_batch)} problems")
                
                
            except Exception as e:
                print(f"Error processing batch {current_batch_num}: {e}")
                continue
        
        # Calculate final accuracies
        final_accuracy = {}
        for strategy, stats in strategy_accuracy.items():
            if stats['total'] > 0:
                final_accuracy[strategy] = stats['correct'] / stats['total']
            else:
                final_accuracy[strategy] = 0.0
        
        return {
            'problem_results': all_results,
            'strategy_accuracy': final_accuracy,
            'total_problems': len(problems),
            'n_gen': n_gen,
            'tail_n': tail_n,
            'max_concurrent_problems': max_concurrent_problems,
            'batch_size': batch_size
        }

async def main():
    print("AIME 2025 Evaluation Script")
    print("="*50)
    
    if not HF_AVAILABLE:
        print("⚠️  Hugging Face datasets not available. Install with: pip install datasets")
        print("Falling back to local JSON file support only.\n")
    
    # Dataset configuration
    print("\nDataset Options:")
    print("1. Hugging Face dataset (e.g., 'AI-MO/aimo-validation-aime', 'hendrycks/competition_math')")
    print("2. Local JSON file path")
    print("3. Press Enter for sample data")
    
    dataset_source = input("\nEnter dataset source: ").strip()
    if not dataset_source:
        dataset_source = "MathArena/aime_2025"
    
    dataset_config = None
    split = "train"
    
    # If it's a HF dataset, ask for additional configuration
    if HF_AVAILABLE and not dataset_source.endswith('.json') and dataset_source != "sample":
        dataset_config = input("Enter dataset configuration (optional, press Enter to skip): ").strip()
        if not dataset_config:
            dataset_config = None
            
        split_input = input("Enter dataset split (default: test): ").strip()
        if split_input:
            split = split_input
    
    # Generation configuration
    try:
        n_gen = int(input("Enter number of generations per problem (default 5): ") or "5")
    except ValueError:
        n_gen = 5
    
    try:
        tail_n = int(input("Enter tail confidence window (default 2048): ") or "2048")
    except ValueError:
        tail_n = 2048
    
    # Concurrency configuration
    print("\n🚀 Performance Configuration:")
    try:
        max_concurrent = int(input("Max concurrent problems (default 3, higher = faster but more API load): ") or "3")
    except ValueError:
        max_concurrent = 5
    
    try:
        batch_size = int(input("Batch size for processing (default 10): ") or "10")
    except ValueError:
        batch_size = 10
    
    print(f"\n⚡ Performance Settings:")
    print(f"  • {max_concurrent} problems will be processed simultaneously")
    print(f"  • Processing in batches of {batch_size} problems")
    print(f"  • Each problem generates {n_gen} responses concurrently")
    print(f"  • Total concurrent API calls per batch: up to {max_concurrent * n_gen}")
    
    # Initialize client
    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
        timeout=30 * 60,
    )
    
    # Get model
    models = await client.models.list()
    model_id = models.data[0].id
    print(f"Using model: {model_id}")
    
    # Initialize evaluator
    evaluator = AIMEEvaluator(client, model_id)
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING AIME 2025 TASK PERFORMANCE EVALUATION")
    print("Goal: Measure how many problems each strategy solves correctly")
    print("="*80)
    
    import time
    start_time = time.time()
    
    results = await evaluator.evaluate_dataset(
        dataset_source=dataset_source, 
        n_gen=n_gen, 
        tail_n=tail_n,
        dataset_config=dataset_config,
        split=split,
        max_concurrent_problems=max_concurrent,
        batch_size=batch_size
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate task-level metrics
    total_generations = sum(pr.get('total_generated', 0) for pr in results['problem_results'])
    total_correct_generations = sum(pr.get('n_correct', 0) for pr in results['problem_results'])
    generation_accuracy = total_correct_generations/total_generations if total_generations > 0 else 0.0
    
    # Count problems that had at least one correct generation (solvability)
    problems_with_correct_solutions = sum(1 for pr in results['problem_results'] if pr.get('n_correct', 0) > 0)
    solvability_rate = problems_with_correct_solutions / results['total_problems'] if results['total_problems'] > 0 else 0.0

    # Print final results
    print("\n" + "="*80)
    print("FINAL AIME 2025 TASK PERFORMANCE RESULTS")
    print("="*80)
    print(f"Dataset source: {dataset_source}")
    if dataset_config:
        print(f"Dataset config: {dataset_config}")
    print(f"Dataset split: {split}")
    print(f"Total problems in task: {results['total_problems']}")
    print(f"Generations per problem: {results['n_gen']}")
    print(f"Tail confidence window: {results['tail_n']}")
    print(f"Max concurrent problems: {results['max_concurrent_problems']}")
    print(f"Batch size: {results['batch_size']}")
    print(f"⏱️  Total execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
    if results['total_problems'] > 0:
        print(f"⚡ Average time per problem: {execution_time/results['total_problems']:.1f} seconds")
    print()
    
    print("📊 TASK ACCURACY - Problems Solved Correctly by Strategy:")
    print("="*60)
    
    # Get strategy results with counts
    strategy_results = []
    for strategy_name, accuracy in results['strategy_accuracy'].items():
        problems_solved = int(accuracy * results['total_problems']) if results['total_problems'] > 0 else 0
        strategy_results.append((strategy_name, accuracy, problems_solved))
    
    # Sort by accuracy for better display
    strategy_results.sort(key=lambda x: x[1], reverse=True)
    
    print("Standard Confidence Strategies:")
    for strategy_name, accuracy, problems_solved in strategy_results:
        if strategy_name in ['random', 'highest_confidence', 'lowest_confidence', 'oracle']:
            print(f"  {strategy_name.replace('_', ' ').title():<20}: {problems_solved:>2}/{results['total_problems']:<2} = {accuracy:.2%}")
    
    print("\nExponential Confidence Strategies:")
    for strategy_name, accuracy, problems_solved in strategy_results:
        if strategy_name in ['highest_exp_confidence', 'lowest_exp_confidence']:
            print(f"  {strategy_name.replace('_', ' ').title():<20}: {problems_solved:>2}/{results['total_problems']:<2} = {accuracy:.2%}")
    
    # Find best performing strategies
    best_strategy = max(strategy_results, key=lambda x: x[1])
    print(f"\n🏆 Best Strategy: {best_strategy[0].replace('_', ' ').title()} - {best_strategy[2]}/{results['total_problems']} problems ({best_strategy[1]:.2%})")
    
    print("\n📈 Generation-Level Analysis (Supplementary):")
    print("="*60)
    print(f"  Problems with ≥1 correct solution: {problems_with_correct_solutions}/{results['total_problems']} ({solvability_rate:.2%})")
    print(f"  Total generations produced: {total_generations}")
    print(f"  Total correct generations: {total_correct_generations}")
    print(f"  Generation-level accuracy: {generation_accuracy:.2%}")
    
    # Performance metrics
    if total_generations > 0:
        print(f"  Generations per second: {total_generations/execution_time:.1f}")
        print(f"  API efficiency: {total_generations} total calls in {execution_time:.1f}s")
    
    # Save results with comprehensive metrics
    dataset_name = dataset_source.replace('/', '_').replace('.json', '')
    output_file = f"aime_results_{dataset_name}_n{n_gen}_c{max_concurrent}_b{batch_size}_tail{tail_n}.json"
    
    # Calculate strategy performance details
    strategy_details = {}
    for strategy_name, accuracy in results['strategy_accuracy'].items():
        problems_solved = int(accuracy * results['total_problems']) if results['total_problems'] > 0 else 0
        strategy_details[strategy_name] = {
            'problems_solved': problems_solved,
            'total_problems': results['total_problems'],
            'task_accuracy': accuracy,
            'strategy_type': 'exponential_confidence' if 'exp_confidence' in strategy_name else 'standard_confidence'
        }
    
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format with task-focused metrics
        json_results = {
            'dataset_info': {
                'source': dataset_source,
                'config': dataset_config,
                'split': split,
                'total_problems': results['total_problems']
            },
            'experiment_config': {
                'n_gen': results['n_gen'],
                'tail_n': results['tail_n'],
                'max_concurrent_problems': results['max_concurrent_problems'],
                'batch_size': results['batch_size'],
                'execution_time_seconds': execution_time
            },
            'task_performance': {
                'strategy_details': strategy_details,
                'best_strategy': {
                    'name': best_strategy[0],
                    'problems_solved': best_strategy[2],
                    'task_accuracy': best_strategy[1]
                },
                'solvability_analysis': {
                    'problems_with_correct_solutions': problems_with_correct_solutions,
                    'solvability_rate': solvability_rate
                }
            },
            'generation_analysis': {
                'total_api_calls': total_generations,
                'total_correct_generations': total_correct_generations,
                'generation_accuracy': generation_accuracy,
                'calls_per_second': total_generations/execution_time if execution_time > 0 else 0,
                'avg_time_per_problem': execution_time/results['total_problems'] if results['total_problems'] > 0 else 0
            }
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print some popular dataset suggestions for future use
    print("\n" + "="*80)
    print("POPULAR AIME/MATH DATASETS ON HUGGING FACE:")
    print("="*80)
    print("• AI-MO/aimo-validation-aime - AIME validation problems")
    print("• hendrycks/competition_math - Competition mathematics problems")
    print("• lighteval/MATH - Mathematical reasoning dataset")
    print("• AI-MO/NuminaMath-CoT - Mathematical problems with chain-of-thought")
    print("• microsoft/orca-math-word-problems-200k - Mathematical word problems")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())