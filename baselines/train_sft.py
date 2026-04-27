#!/usr/bin/env python3
import argparse
from html import parser
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from datasets import Dataset as HFDataset
from collections import defaultdict
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from profiler import FlopsTimingCallback
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from math_evaluator import MATHLoader, AIMELoader

DATA_DIR = "<path_to_generated_data>"

os.environ["WANDB_PROJECT"] = ''
os.environ['WANDB_API_KEY'] = ''


CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

def seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def cleanup_model(*models):
    """Aggressively free GPU memory."""
    for m in models:
        if m is not None:
            m.cpu()
            del m
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.memory_reserved()/1e9:.1f}GB reserved")
    
def split_train_val(data, seed, val_ratio=0.25):
    """
    Split a list of JSON objects into train and validation sets based on unique problem_ids.
    Each object must contain a 'problem_id' field. - Function adapted from code in scatr that takes df.
    """
    seed_everywhere(seed)

    # Extract unique problem_ids
    problem_ids = sorted(np.array(list({item['problem_id'] for item in data})))
    
    # Shuffle safely
    shuffled_ids = np.random.permutation(problem_ids)

    # Split
    val_size = int(len(shuffled_ids) * val_ratio)
    val_problem_ids = set(shuffled_ids[:val_size])
    train_problem_ids = set(shuffled_ids[val_size:])

    # Filter data
    train_data = [item for item in data if item['problem_id'] in train_problem_ids]
    val_data = [item for item in data if item['problem_id'] in val_problem_ids]

    return train_data, val_data

def ensure_chat_template(tokenizer):
    """Set ChatML fallback if the tokenizer has no chat template."""
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHATML_TEMPLATE


# ── Custom data collator with label masking ───────────────────────────────────

@dataclass
class CompletionOnlyCollator:
    """Pad input_ids / attention_mask / labels.  Labels use -100 for padding."""
    tokenizer: object

    def __call__(self, features: List[Dict]) -> Dict:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            length = len(f["input_ids"])
            pad_len = max_len - length
            # Left-pad so the final tokens (the ones we care about) stay at the end
            batch["input_ids"].append([pad_id] * pad_len + f["input_ids"])
            batch["attention_mask"].append([0] * pad_len + f["attention_mask"])
            batch["labels"].append([-100] * pad_len + f["labels"])
        return {
            k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()
        }


# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file, skipping malformed lines."""
    data = []
    skipped = 0
    with open(filepath, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"WARNING: skipping malformed JSON in {filepath} line {i}: {e}")
    if skipped:
        print(f"  → loaded {len(data)} records, skipped {skipped}")
    return data


def load_data(model_name: str, dataset_name: str, turn: int) -> List[Dict]:
    """Load response metrics and join with problem text."""
    filepath = os.path.join(
        DATA_DIR,
        f"{model_name}/{dataset_name}/turn{turn}/all_response_metrics.jsonl",
    )
    res = load_jsonl(filepath)

    repo_root = Path(__file__).resolve().parent.parent

    dataset_name_to_problems = {
        "humaneval": str(repo_root / "datasets" / "humaneval.jsonl"),
        "kodcode":   str(repo_root / "datasets" / "kodcode_1000.jsonl"),
        "math500":   'HuggingFaceH4/MATH-500',
        "aime":      'MathArena/aime_2025',
        "aime24": 'Maxwell-Jia/AIME_2024',
    }

    problems_path = dataset_name_to_problems[dataset_name]

    # Build problem_dict: problem_id -> problem_text
    if dataset_name in ("humaneval", "kodcode", "bigcodebench_hard"):
        problems_raw  = load_jsonl(problems_path)
        problem_dict = {p["task_id"]: p["prompt"] for p in problems_raw}
    elif dataset_name == "math500":
        loader = MATHLoader()
        math_problems = loader.load(problems_path, split='test')
        problem_dict = {p.problem_id: p.problem_text for p in math_problems}
    elif dataset_name == "aime":
        loader = AIMELoader()
        aime_problems = loader.load(problems_path, split='train')
        problem_dict = {p.problem_id: p.problem_text for p in aime_problems}
    elif dataset_name == "aime24":
        loader = AIMELoader()
        aime24_problems = loader.load(problems_path, split='train')
        problem_dict = {p.problem_id: p.problem_text for p in aime24_problems}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    for item in res:
        problem_id = item["problem_id"]
        if dataset_name == "humaneval":
            problem_id = problem_id.replace("_", "/")
        item["problem"] = problem_dict.get(problem_id, "Unknown Problem")

    return res


# ── chat-formatted dataset ───────────────────────────────────────────────────

USER_TEMPLATE = (
    "Determine the correctness of the following answer:\n"
    "Problem: {problem}\n"
    "Answer: {response_text}\n"
    "Is this answer correct (yes/no)?"
)


def build_tokenized_dataset(
    data: List[Dict], tokenizer, max_seq_len: int = 16384
) -> HFDataset:
    """
    Build a HuggingFace Dataset with ``input_ids``, ``attention_mask``, and
    ``labels`` columns.  Labels are -100 for all prompt tokens so loss is only
    computed on the assistant's "yes"/"no" response.
    """
    all_input_ids, all_attn, all_labels = [], [], []

    for item in data:
        label_text = "yes" if item["is_correct"] else "no"
        
        response_text_ids = tokenizer.encode(item["response_text"], add_special_tokens=False)
        if len(response_text_ids) > max_seq_len:
            response_text = tokenizer.decode(response_text_ids[-max_seq_len:], skip_special_tokens=True)
        else:
            response_text = item["response_text"]
            
        if 'gpt' in tokenizer.name_or_path.lower():
            response_text = response_text.replace('<|channel|>final<|message|>', '')  # remove any existing generation prompts to avoid duplication        
            response_text = response_text.replace('<|start|>assistant<|channel|>', '').replace('<|channel|>analysis<|message|>', '')  # remove any existing assistant/analysis tags to avoid duplication
            # remove any <|text|> wehre text is anything (e.g. <|analysis|>, <|final|>, <|message|>, etc.)
            import re
            response_text = re.sub(r'<\|.*?\|>', '', response_text)
            
        user_content = USER_TEMPLATE.format(
            problem=item["problem"],
            response_text=response_text, 
        )

        # Prompt = everything up to (and including) the assistant header
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        # Full = prompt + assistant response + any EOS / closing tokens
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": label_text},
            ],
            tokenize=False, add_generation_prompt=False
        )
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        prompt_len = min(len(prompt_ids), len(full_ids))

        # Labels: -100 for prompt, real ids for completion (shifted by trainer)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        assert len(labels) == len(full_ids)

        all_input_ids.append(full_ids)
        all_attn.append([1] * len(full_ids))
        all_labels.append(labels)

    return HFDataset.from_dict(
        {"input_ids": all_input_ids, "attention_mask": all_attn, "labels": all_labels}
    )

# ── training ──────────────────────────────────────────────────────────────────

def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    model_name: str,
    output_dir: str,
    seed: int,
    bs: int = 1,
    grad_accumulation_steps: int = 16,
    max_seq_len: int = 16384,
    train_ds: str = None,
) -> str:
    """Fine-tune with Trainer.  Loss only on the assistant token."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 
    ensure_chat_template(tokenizer)

    # ── model ──
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── LoRA configuration ──
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,  # scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", 'gate_up_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print(model)  # look for lora_A / lora_B in the tree

    # First, inspect what's inside the experts
    for name, module in model.named_modules():
        if 'expert' in name.lower() or 'mlp' in name.lower():
            print(f"{name}: {type(module).__name__}")
    # Check what parameters (leaf tensors) live inside experts
    for name, param in model.named_parameters():
        if 'mlp' in name:
            print(f"{name}: {param.shape}")
    flops_callback = FlopsTimingCallback(model, output_dir=output_dir, profile_step=5)

    # Log token info
    true_ids = tokenizer.encode("yes", add_special_tokens=False)
    false_ids = tokenizer.encode("no", add_special_tokens=False)
    print(f"'yes'  token ids: {true_ids}")
    print(f"'no' token ids: {false_ids}")

    # ── dataset (pre-tokenized with label masking) ──
    dataset = build_tokenized_dataset(train_data, tokenizer, max_seq_len=max_seq_len)
    val_dataset = build_tokenized_dataset(val_data, tokenizer, max_seq_len=max_seq_len)
    print(f"Training examples: {len(dataset)}, Validation examples: {len(val_dataset)}")
    # Show a decoded sample so we can verify formatting
    sample_ids = dataset[0]["input_ids"]
    sample_labels = dataset[0]["labels"]
    print(f"Sample prompt + response:\n{tokenizer.decode(sample_ids)[:500]}")
    n_train_tokens = sum(1 for l in sample_labels if l != -100)
    print(f"Tokens with loss in first example: {n_train_tokens}")

    collator = CompletionOnlyCollator(tokenizer=tokenizer)
    
    #get max sequence length from train data
    max_seq_len = max(len(f["input_ids"]) for f in dataset)
    print(f"Max sequence length in training data: {max_seq_len}")

    # ── compute expected steps for user info ── 
    num_examples = len(dataset)
    batch_size = bs
    grad_accum = grad_accumulation_steps
    num_epochs = 1 if train_ds != 'aime' else 6  # train longer on the smaller AIME dataset
    steps_per_epoch = math.ceil(num_examples / batch_size) // grad_accum
    total_steps = steps_per_epoch * num_epochs
    print(f"\n{'─' * 50}")
    print(f"Training config:")
    print(f"  Examples:           {num_examples}")
    print(f"  Batch size:         {batch_size} × {grad_accum} grad accum = {batch_size * grad_accum} effective")
    print(f"  Steps per epoch:    {steps_per_epoch}")
    print(f"  Total steps:        {total_steps} ({num_epochs} epochs)")
    print(f"  Checkpoints saved:  {output_dir}/checkpoint-{{step}}")
    print(f"  Final model saved:  {output_dir}")
    print(f"{'─' * 50}\n")
    sys.stdout.flush()

    # ── training config ──
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        logging_strategy="steps",
        #eval_strategy="epoch",
        save_strategy="epoch",
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_loss",
        bf16=True,
        seed=seed,
        report_to="wandb",
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    # ── trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        #eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[flops_callback]
    )
    
    # After building dataset, before trainer.train()
    sample = dataset[0]
    sample_ids = torch.tensor([sample["input_ids"]]).to(model.device)
    sample_mask = torch.tensor([sample["attention_mask"]]).to(model.device)
    flops_callback.measure_with_sample(sample_ids, sample_mask)

    trainer.train()
    print(f"\nTraining complete. Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    sys.stdout.flush()
    
    del trainer
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return output_dir


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_best_of_n(
    hf_model, model_dir: str, test_data: List[Dict], gpu_id: int,
    batch_size: int = 32, max_seq_len: int = 16384
) -> float:
    
    test_data = test_data[:100]  # TEMP: limit to 100 samples for faster evaluation during development
    
    print(f"Evaluating best-of-N on GPU {gpu_id} with model {model_dir}...")
    print(f"Model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer.padding_side = "left"  # critical for causal LM batching
    ensure_chat_template(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(
        hf_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Base model {hf_model} loaded for evaluation.")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    true_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("no", add_special_tokens=False)[0]

    # ── build ALL prompts upfront ──────────────────────────────────────────
    all_prompts = []
    all_items   = []
    for item in test_data:
        response_text_ids = tokenizer.encode(item["response_text"], add_special_tokens=False)
        if len(response_text_ids) > max_seq_len:
            response_text = tokenizer.decode(response_text_ids[-max_seq_len:], skip_special_tokens=True)
        else:
            response_text = item["response_text"]
            
        messages = [{"role": "user", "content": USER_TEMPLATE.format(
            problem=item["problem"],
            response_text=response_text,
        )}]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if 'qwen' in hf_model.lower():
            prompt += '<think>\n\n</think>\n\n'
        
        if 'gpt' in hf_model.lower():
            prompt += '<|channel|>final<|message|>'
        
        all_prompts.append(prompt)
        all_items.append(item)
    print(f"Built {len(all_prompts)} prompts for evaluation.")
    # ── batched forward passes ─────────────────────────────────────────────
    all_true_probs  = []
    all_false_probs = []

    import time
    infer_times = []
    start_time = time.time() 
    with torch.no_grad():
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Evaluating"):
            batch_prompts = all_prompts[i : i + batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,          # left-pad to same length
                truncation=True,
            ).to(model.device)

            # logits[:, -1, :] = next-token distribution for every item in batch
            start_infer = time.time()
            logits = model(**inputs).logits[:, -1, :]
            probs  = torch.softmax(logits, dim=-1)
            end_infer = time.time()
            infer_times.append(end_infer - start_infer)
            
            all_true_probs.extend(probs[:, true_token_id].cpu().tolist())
            all_false_probs.extend(probs[:, false_token_id].cpu().tolist())
    end_time = time.time()
    print(f"Completed forward passes for evaluation. Collected probabilities for {len(all_true_probs)} samples.")
    print(f"Total evaluation time: {end_time - start_time:.2f} seconds")
    print(f"Average inference time per sample: {sum(infer_times) / len(infer_times):.4f} seconds")
    # ── per-rollout predictions ────────────────────────────────────────────
    rollout_predictions = []
    for idx, item in enumerate(all_items):
        rollout_predictions.append({
            "problem_id":      item["problem_id"],
            "rollout_idx":     item.get("rollout_idx", idx),
            "true_prob":       all_true_probs[idx],
            "false_prob":      all_false_probs[idx],
            "predicted_label": "yes" if all_true_probs[idx] > all_false_probs[idx] else "no",
            "is_correct":      item["is_correct"],
        })

    # ── best-of-N selection ────────────────────────────────────────────────
    problems = defaultdict(list)
    for idx, item in enumerate(all_items):
        problems[item["problem_id"]].append({
            "rollout":         item,
            "rollout_idx":     item.get("rollout_idx", idx),
            "true_prob":       all_true_probs[idx],
            "false_prob":      all_false_probs[idx],
            "predicted_token": true_token_id if all_true_probs[idx] > all_false_probs[idx] else false_token_id,
        })

    correct_predictions = 0
    total_problems      = 0
    problem_results     = []
    for problem_id, rollout_scores in problems.items():
        true_preds = [r for r in rollout_scores if r["predicted_token"] == true_token_id]
        if true_preds:
            best = max(true_preds, key=lambda x: x["true_prob"])
        else:
            best = min(rollout_scores, key=lambda x: x["false_prob"])

        is_correct = best["rollout"]["is_correct"]
        if is_correct:
            correct_predictions += 1
        total_problems += 1

        problem_results.append({
            "problem_id":      problem_id,
            "best_rollout_idx": best["rollout_idx"],
            "best_true_prob":  best["true_prob"],
            "best_false_prob": best["false_prob"],
            "predicted_label": "yes" if best["predicted_token"] == true_token_id else "no",
            "is_correct":      is_correct,
            "n_rollouts":      len(rollout_scores),
        })

    accuracy = correct_predictions / total_problems if total_problems > 0 else 0.0
    print(f"\nEvaluation complete. {correct_predictions}/{total_problems} = {accuracy:.2%}")

    # ── cleanup ───────────────────────────────────────────────────────────
    cleanup_model(model, base_model)
    del tokenizer

    return accuracy, rollout_predictions, problem_results, end_time - start_time, infer_times

# ── per-seed entry-point ─────────────────────────────────────────────────────

def process_seed(args_tuple):
    """Process a single (seed, turn) experiment."""
    seed, gpu_id, train_ds, test_ds, model_name, hf_model, turn, bs, bs_eval, grad_accumulation_steps, max_seq_len, eval_only, inference_estimate = args_tuple
    print(f"Eval: {eval_only}")
    try:
        print(f"\nSeed {seed}, Turn {turn} (GPU {gpu_id}): Starting...")

        train_data = load_data(model_name, train_ds, turn)
        test_data = load_data(model_name, test_ds, turn)

        # Split training data into train/val by problem_id
        train_split, val_split = split_train_val(train_data, seed)

        print(
            f"Seed {seed}: {len(train_split)} train / {len(val_split)} val / {len(test_data)} test samples"
        )
        
        #print distributions of correct/incorrect in train/val/test
        def print_distribution(name, data):
            correct = sum(1 for d in data if d["is_correct"])
            incorrect = len(data) - correct
            print(f"{name} distribution: {correct} correct / {incorrect} incorrect ({correct/len(data):.2%} correct)")
        
        print_distribution("Train", train_split)
        print_distribution("Validation", val_split)
        print_distribution("Test", test_data)

        #return seed, None, None  # TEMP: skip training/eval to test data loading and distribution printing
        
        output_dir = f"/tmp/scatr/models/baselines/{model_name}_{train_ds}_{test_ds}/seed_{seed}_turn{turn}"
        
        

        if eval_only:
            print(f"Seed {seed}: Eval-only mode, skipping training and using existing model at {output_dir}")
            model_dir = output_dir
            print(model_dir)
        else:
            print(f"Seed {seed}: Training...")
            model_dir = train_model(train_split, val_split, hf_model, output_dir, seed, bs, grad_accumulation_steps, max_seq_len=max_seq_len, train_ds=train_ds)

        if inference_estimate:
            output_dir = f"/tmp/scatr/models/inference_estimates_v2/{model_name}_{train_ds}_{test_ds}/seed_{seed}_turn{turn}"
        print(f"Seed {seed}: Evaluating...")
        import time
        eval_start_time = time.time()
        accuracy, rollout_predictions, problem_results, eval_time, infer_times = evaluate_best_of_n(hf_model, model_dir, test_data, gpu_id, batch_size=bs_eval, max_seq_len=max_seq_len)
        eval_end_time = time.time()
        print(f"Seed {seed}: Accuracy = {accuracy:.4f}")

        # log dir mirrors the model dir
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if inference_estimate:
            # If this is an inference time estimation run, log the timing results instead of accuracy
            timing_log_path = log_dir / f"inference_timing_seed{seed}_turn{turn}.json"
            with open(timing_log_path, "w") as f:
                json.dump({
                    "model_dir": model_dir,
                    "train_ds": train_ds,
                    "test_ds": test_ds,
                    "seed": seed,
                    "turn": turn,
                    "eval_bon_selection": eval_end_time - eval_start_time,
                    "infer_times_seconds": infer_times,
                    "eval_time_seconds": eval_time,
                    
                }, f, indent=2)
            print(f"Logged inference timing → {timing_log_path}")
            return seed, None, None

        # per-rollout probabilities
        rollout_log_path = log_dir / f"eval_rollouts_seed{seed}_turn{turn}.jsonl"
        with open(rollout_log_path, "w") as f:
            for row in rollout_predictions:
                f.write(json.dumps(row) + "\n")

        # per-problem best-of-N results + overall accuracy
        results_log_path = log_dir / f"eval_results_seed{seed}_turn{turn}.json"
        with open(results_log_path, "w") as f:
            json.dump({
                "model_dir":   model_dir,
                "train_ds":    train_ds,
                "test_ds":     test_ds,
                "seed":        seed,
                "turn":        turn,
                "accuracy":    accuracy,
                "n_correct":   int(accuracy * len(problem_results)),
                "n_problems":  len(problem_results),
                "problems":    problem_results,
            }, f, indent=2)

        print(f"Logged rollouts → {rollout_log_path}")
        print(f"Logged results  → {results_log_path}")
    except Exception as e:
        print(f"Seed {seed}: Error – {e}")
        return seed, None, str(e)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a correctness verifier with LoRA and evaluate best-of-N"
    )
    parser.add_argument("--train", type=str, required=True, help="train dataset name")
    parser.add_argument("--test", type=str, required=True, help="test dataset name")
    parser.add_argument("--model", type=str, default=None, help="model short name or HF id")
    parser.add_argument("--bs", type=int, default=1, help="per-device batch size for training")
    parser.add_argument("--bs_eval", type=int, default=16, help="per-device batch size for evaluation")
    parser.add_argument("--grad_accum", type=int, default=16, help="gradient accumulation steps for training")
    parser.add_argument("--max_seq_len", type=int, default=16384, help="maximum sequence length for training and evaluation")
    parser.add_argument("--eval_only", action="store_true", help="skip training and only run evaluation (expects existing model checkpoints)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--turn", type=int, default=None)
    parser.add_argument("--inference_estimate", action="store_true", help="estimate inference time")
    parser.add_argument(
        "--sequential", action="store_true", help="run seeds sequentially"
    )
    args = parser.parse_args()

    seeds = [32, 42, 52]
    turns = [1, 2, 3]

    model_mapping = {
        "gptoss": "openai/gpt-oss-20b",
        "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
        "qwen1_7b": "Qwen/Qwen3-1.7B",
        "qwen30b": "Qwen/Qwen3-30B-A3B"
    }
    args.hf_model = model_mapping.get(args.model, args.model)

    print(f"Train: {args.train}")
    print(f"Test:  {args.test}")
    print(f"Model: {args.model}  →  {args.hf_model}")
    print(f"Seeds: {seeds}")
    print("=" * 70)
    
    if args.seed is not None and args.turn is not None:
        task = (args.seed, 0, args.train, args.test, args.model, args.hf_model,
                args.turn, args.bs, args.bs_eval, args.grad_accum, args.max_seq_len, args.eval_only, args.inference_estimate)
        process_seed(task)
        return

    tasks = []
    for idx, seed in enumerate(seeds):
        for turn in turns:
            tasks.append(
                (seed, 0, args.train, args.test, args.model, args.hf_model, turn, args.bs, args.bs_eval, args.grad_accum, args.max_seq_len, args.eval_only, args.inference_estimate)
            )

    results = []
    if args.sequential:
        for task in tasks:
            seed, gpu_id, train_ds, test_ds, model_name, hf_model, turn, bs, bs_eval, grad_accum, max_seq_len, eval_only, inference_estimate = task
            
            cmd = [
                sys.executable, __file__,
                "--train", train_ds,
                "--test", test_ds,
                "--model", model_name,
                "--bs", str(bs),
                "--bs_eval", str(bs_eval),
                "--grad_accum", str(grad_accum),
                "--max_seq_len", str(max_seq_len),
                "--sequential",
                "--seed", str(seed),
                "--turn", str(turn),
            ]
            if eval_only:
                cmd.append("--eval_only")
            
            print(f"Launching subprocess: seed={seed}, turn={turn}")    
            subprocess.run(cmd, check=True)
    else:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        print(f"\nParallel processing with {args.num_gpus} GPUs...")
        with mp.Pool(processes=min(len(tasks), args.num_gpus)) as pool:
            results = pool.map(process_seed, tasks)

    # ── report ──
    print(f"\n{'=' * 70}")
    print("DONE. Summary of results:")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
