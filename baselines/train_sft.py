#!/usr/bin/env python3
"""
Train a model to detect whether an answer is correct.
Uses SFT with loss only on the final classification token.
Evaluation uses best-of-N selection based on "true" token probability.
"""

import argparse
from copy import Error
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from collections import defaultdict
from tqdm import tqdm

DATA_DIR = "/efs/cactts/data"


class CorrectnessDataset(Dataset):
    """Dataset for training correctness detection."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        response_text = item['response_text']
        is_correct = item['is_correct']
        problem = item['problem']
        
        label_text = "true" if is_correct else "false"
        prompt = f"""Evaluate whether this answer is correct.
    Problem: {problem}
    Answer: {response_text}
    This answer is """
        full_text = f"{prompt}{label_text}"
        
        # Tokenize full text (this is ground truth)
        full_tokens = self.tokenizer(
            full_text, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            truncation=True
        )
        
        # Tokenize ONLY the prompt with special tokens to see where it ends
        prompt_tokens = self.tokenizer(
            prompt, 
            add_special_tokens=True,
            truncation=False  # Don't truncate, we need accurate length
        )
        
        input_ids = full_tokens['input_ids']
        attention_mask = full_tokens['attention_mask']
        
        # The prompt length in full_tokens is len(prompt_tokens) - 1 because:
        # prompt_tokens: [BOS, ...prompt..., EOS]
        # full_tokens:   [BOS, ...prompt..., ...label..., EOS]
        # So we exclude the EOS from prompt_tokens count
        prompt_length = len(prompt_tokens['input_ids']) - 1
        
        # Create labels: -100 for prompt (no loss), actual tokens for label+EOS
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    return data


def load_data(model_name, dataset_name, turn):
    filepath = os.path.join(DATA_DIR, f"{model_name}/{dataset_name}/turn{turn}/all_response_metrics.jsonl")
    res = load_jsonl(filepath)
    
    # join with problem text from different directory
    dataset_name_to_problems = {
        'humaneval': "/home/ubuntu/cactts/datasets/humaneval.jsonl",
        'kodcode': "/home/ubuntu/cactts/datasets/kodcode_1000.jsonl",
    }

    problems = load_jsonl(dataset_name_to_problems[dataset_name])
    problem_dict = {p['task_id']: p['prompt'] for p in problems}
    
    # Update each item in res with problem text
    for item in res:
        if dataset_name == 'humaneval':
            problem_id = item['problem_id'].replace('_', '/')
        item['problem'] = problem_dict.get(problem_id, "Unknown Problem")
    
    return res

def train_model(train_data: List[Dict], model_name: str, output_dir: str, 
                seed: int, gpu_id: int):
    """Train the model on the training data."""
    
    # Set environment variable for GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Create dataset
    train_dataset = CorrectnessDataset(train_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        bf16=False,
        fp16=True,
        seed=seed,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir


def evaluate_best_of_n(model_dir: str, test_data: List[Dict], gpu_id: int) -> float:
    """
    Evaluate using best-of-N selection.
    For each problem, select rollout with highest probability of "true" token.
    If no rollout predicts "true", select the one with lowest probability of "false".
    """
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Get token IDs for "true" and "false"
    true_token_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("false", add_special_tokens=False)[0]
    
    # Group by problem_id
    problems = defaultdict(list)
    for item in test_data:
        problems[item['problem_id']].append(item)
    
    correct_predictions = 0
    total_problems = 0
    
    with torch.no_grad():
        for problem_id, rollouts in tqdm(problems.items(), desc="Evaluating"):
            rollout_scores = []
            
            # Evaluate each rollout for this problem
            for rollout in rollouts:
                response_text = rollout['response_text']
                problem = rollout['problem']
                prompt = f"""Evaluate whether this answer is correct. Write 'This answer is true' or 'This answer is false'.
Problem: {problem}
Answer: {response_text}
This answer is """
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Get logits for next token
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get probabilities of both tokens
                true_prob = probs[true_token_id].item()
                false_prob = probs[false_token_id].item()
                
                # Determine which token is predicted (argmax)
                predicted_token = true_token_id if true_prob > false_prob else false_token_id
                
                rollout_scores.append({
                    'rollout': rollout,
                    'true_prob': true_prob,
                    'false_prob': false_prob,
                    'predicted_token': predicted_token
                })
            
            # First, try to find rollouts that predict "true"
            true_predictions = [r for r in rollout_scores if r['predicted_token'] == true_token_id]
            
            if true_predictions:
                # Select the one with highest probability of "true"
                best = max(true_predictions, key=lambda x: x['true_prob'])
            else:
                # All predict "false", select the one with lowest probability of "false"
                best = min(rollout_scores, key=lambda x: x['false_prob'])
            
            # Check if the selected rollout is correct
            if best['rollout']['is_correct']:
                correct_predictions += 1
            
            total_problems += 1
    
    accuracy = correct_predictions / total_problems if total_problems > 0 else 0.0
    return accuracy


def process_seed(args_tuple):
    """Process a single seed (for parallel execution)."""
    seed, gpu_id, train_data, test_data, model_name, hf_model, turn = args_tuple
    
    try:
        print(f"\nSeed {seed} (GPU {gpu_id}): Starting...")
        
        # Load data
        train_data = load_data(model_name, train_data, turn)
        test_data = load_data(model_name, test_data, turn)
        
        print(f"Seed {seed}: Loaded {len(train_data)} train samples, {len(test_data)} test samples")
        
        # Create output directory
        output_dir = f"./sft_output/seed_{seed}"
        
        # Train model
        print(f"Seed {seed}: Training...")
        model_dir = train_model(train_data, hf_model, output_dir, seed, gpu_id)
        
        # Evaluate
        print(f"Seed {seed}: Evaluating...")
        accuracy = evaluate_best_of_n(model_dir, test_data, gpu_id)
        
        print(f"Seed {seed}: Accuracy = {accuracy:.4f}")
        
        return seed, accuracy, None
        
    except Exception as e:
        print(f"Seed {seed}: Error - {str(e)}")
        return seed, None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Finetune model with SFT and evaluate with best-of-N'
    )
    parser.add_argument('--train', type=str, required=True, 
                       help='train dataset')
    parser.add_argument('--test', type=str, required=True, 
                       help='test dataset (humaneval or kodcode)')
    parser.add_argument('--model', type=str, default=None,
                       help='Base model to fine-tune')
    parser.add_argument('--num-gpus', type=int, default=3, 
                       help='Number of GPUs to use')
    parser.add_argument('--sequential', action='store_true', 
                       help='Run sequentially instead of parallel')
    args = parser.parse_args()
    
    seeds = [42, 52, 62]
    turns = [1, 2, 3]
    
    model_mapping = {
        "gptoss": "openai/gpt-oss-20b",
        "ollmo7b": "allenai/OLMo-2-1124-7B-Instruct",
        "qwen1_7b": "Qwen/Qwen3-1.7B",
    }
    
    args.hf_model = model_mapping.get(args.model, args.model) 
    
    print(f"Train: {args.train}")
    print(f"Test: {args.test}")
    print(f"Model: {args.model}")
    print(f"HF Model: {args.hf_model}")
    print(f"Seeds: {seeds}")
    print(f"GPUs: {args.num_gpus}")
    print("="*70)
    
    # Create tasks
    tasks = []
    for idx, seed in enumerate(seeds):
        for turn in turns:
            gpu_id = idx % args.num_gpus
            tasks.append((seed, gpu_id, args.train, args.test, args.model, args.hf_model, turn))
        
    # Run training and evaluation
    results = []
    if args.sequential:
        for task in tasks:
            result = process_seed(task)
            results.append(result)
    else:
        import multiprocessing as mp
        print(f"\nStarting parallel processing with {args.num_gpus} GPUs...")
        with mp.Pool(processes=min(len(tasks), args.num_gpus)) as pool:
            results = pool.map(process_seed, tasks)
    
    # Process results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    accuracies = []
    for seed, accuracy, error in results:
        if error:
            print(f"Seed {seed}: Error - {error}")
        else:
            print(f"Seed {seed}: Accuracy = {accuracy:.4f}")
            accuracies.append(accuracy)
    
    # Compute statistics
    if len(accuracies) > 0:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Best-of-N Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Total experiments: {len(seeds)}")
        
        # Save results
        train_name = Path(args.train).stem
        test_name = Path(args.test).stem
        output_file = f"results_sft_{train_name}_{test_name}.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"Finetuned SFT Results\n")
            f.write(f"{'='*70}\n")
            f.write(f"Train: {args.train}\n")
            f.write(f"Test: {args.test}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Seeds: {seeds}\n")
            f.write(f"\n")
            f.write(f"Best-of-N Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"\nIndividual results:\n")
            for seed, accuracy, error in results:
                if error:
                    f.write(f"  Seed {seed}: Error - {error}\n")
                else:
                    f.write(f"  Seed {seed}: {accuracy:.4f}\n")
            f.write(f"{'='*70}\n")
        
        print(f"\nResults saved to {output_file}")
    else:
        print("\nNo successful experiments completed.")


if __name__ == "__main__":
    main()
