#!/usr/bin/env python3
"""
Train a correctness verifier with LoRA-based parameter-efficient fine-tuning.

Approach – next-token prediction on classification tokens:
  User  :  Problem + generated answer + "Is this answer correct?"
  Asst  :  "true" | "false"

Cross-entropy loss is computed **only** on the "true"/"false" assistant tokens
via label masking (labels=-100 for all prompt tokens).  At eval time the model
scores each rollout by P("true") and we pick the best-of-N.
"""

import argparse
import json
import math
import os
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import split_train_val
from profiler import FlopsTimingCallback

DATA_DIR = None  # Set via command-line argument


# ── ChatML fallback (for tokenizers without a built-in chat template) ─────────
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


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

    # Resolve dataset paths relative to the repo root (parent of baselines/)
    repo_root = Path(__file__).resolve().parent.parent

    dataset_name_to_problems = {
        "humaneval": str(repo_root / "datasets" / "humaneval.jsonl"),
        "kodcode": str(repo_root / "datasets" / "kodcode_1000.jsonl"),
    }

    problems = load_jsonl(dataset_name_to_problems[dataset_name])
    problem_dict = {p["task_id"]: p["prompt"] for p in problems}

    for item in res:
        problem_id = item["problem_id"]
        if dataset_name == "humaneval":
            problem_id = problem_id.replace("_", "/")
        item["problem"] = problem_dict.get(problem_id, "Unknown Problem")

    return res


# ── chat-formatted dataset ───────────────────────────────────────────────────

USER_TEMPLATE = (
    "Problem: {problem}\n"
    "Answer: {response_text}\n"
    "Is this answer correct? Respond with true or false."
)


def build_tokenized_dataset(
    data: List[Dict], tokenizer, max_seq_len: int = 16384
) -> HFDataset:
    """
    Build a HuggingFace Dataset with ``input_ids``, ``attention_mask``, and
    ``labels`` columns.  Labels are -100 for all prompt tokens so loss is only
    computed on the assistant's "true"/"false" response.
    """
    all_input_ids, all_attn, all_labels = [], [], []

    for item in data:
        label_text = "true" if item["is_correct"] else "false"
        
        response_text_ids = tokenizer.encode(item["response_text"], add_special_tokens=False)
        if len(response_text_ids) > max_seq_len:
            response_text = tokenizer.decode(response_text_ids[-max_seq_len:], skip_special_tokens=True)
        else:
            response_text = item["response_text"]
        
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
        
        full_text = full_text.replace('assistant\n<think>\n\n</think>\n', f'assistant') # remove thinking tags

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
) -> str:
    """Fine-tune with Trainer.  Loss only on the assistant token."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # match left-pad in collator
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    flops_callback = FlopsTimingCallback(model, output_dir=output_dir, profile_step=5)

    # Log token info
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    print(f"'true'  token ids: {true_ids}")
    print(f"'false' token ids: {false_ids}")

    # ── dataset (pre-tokenized with label masking) ──
    dataset = build_tokenized_dataset(train_data, tokenizer, max_seq_len=16384)
    val_dataset = build_tokenized_dataset(val_data, tokenizer, max_seq_len=16384)
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
    
    # Measure FLOPs with a sample batch for accurate profiling
    sample_input_ids = torch.tensor([dataset[0]["input_ids"]], device=model.device)
    sample_attention_mask = torch.tensor([dataset[0]["attention_mask"]], device=model.device)
    flops_callback.measure_with_sample(sample_input_ids, sample_attention_mask)

    # ── compute expected steps for user info ── 
    num_examples = len(dataset)
    batch_size = bs
    grad_accum = grad_accumulation_steps
    num_epochs = 1
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
        report_to="none",
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
    batch_size: int = 32,
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"  # critical for causal LM batching
    ensure_chat_template(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        hf_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    true_token_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("false", add_special_tokens=False)[0]

    # ── build ALL prompts upfront ──────────────────────────────────────────
    all_prompts = []
    all_items   = []
    for item in test_data:
        messages = [{"role": "user", "content": USER_TEMPLATE.format(
            problem=item["problem"],
            response_text=item["response_text"],
        )}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if 'gpt' in hf_model.lower():
            prompt += '<|channel|>final<|message|>'
        
        all_prompts.append(prompt)
        all_items.append(item)

    # ── batched forward passes ─────────────────────────────────────────────
    all_true_probs  = []
    all_false_probs = []

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
            logits = model(**inputs).logits[:, -1, :]
            probs  = torch.softmax(logits, dim=-1)

            all_true_probs.extend(probs[:, true_token_id].cpu().tolist())
            all_false_probs.extend(probs[:, false_token_id].cpu().tolist())

    # ── best-of-N selection ────────────────────────────────────────────────
    # Re-group by problem_id using indices
    problems = defaultdict(list)
    for idx, item in enumerate(all_items):
        problems[item["problem_id"]].append({
            "rollout":    item,
            "true_prob":  all_true_probs[idx],
            "false_prob": all_false_probs[idx],
            "predicted_token": true_token_id if all_true_probs[idx] > all_false_probs[idx] else false_token_id,
        })

    correct_predictions = 0
    total_problems      = 0
    for problem_id, rollout_scores in problems.items():
        true_preds = [r for r in rollout_scores if r["predicted_token"] == true_token_id]
        if true_preds:
            best = max(true_preds, key=lambda x: x["true_prob"])
        else:
            best = min(rollout_scores, key=lambda x: x["false_prob"])

        if best["rollout"]["is_correct"]:
            correct_predictions += 1
        total_problems += 1

    print(f"\nEvaluation complete. {correct_predictions}/{total_problems} = {correct_predictions/total_problems:.2%}")
    return correct_predictions / total_problems if total_problems > 0 else 0.0


# ── per-seed entry-point ─────────────────────────────────────────────────────

def process_seed(args_tuple):
    """Process a single (seed, turn) experiment."""
    seed, gpu_id, train_ds, test_ds, model_name, hf_model, turn, bs, grad_accumulation_steps = args_tuple

    try:
        print(f"\nSeed {seed}, Turn {turn} (GPU {gpu_id}): Starting...")

        train_data = load_data(model_name, train_ds, turn)
        test_data = load_data(model_name, test_ds, turn)

        # Split training data into train/val by problem_id
        train_split, val_split = split_train_val(train_data, val_ratio=0.2, random_seed=seed)
        ############### FOR NOW ################################
        train_split = train_data  # TEMP: use all data for training and don't do validation and early stopping
        #########################################################
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
        
        output_dir = f"./sft_output/{model_name}_{train_ds}_{test_ds}/seed_{seed}_turn{turn}"
    
        print(f"Seed {seed}: Training...")
        model_dir = train_model(train_split, val_split, hf_model, output_dir, seed, bs, grad_accumulation_steps)

        print(f"Seed {seed}: Evaluating...")
        accuracy = evaluate_best_of_n(hf_model, model_dir, test_data, gpu_id, batch_size=bs)
        print(f"Seed {seed}: Accuracy = {accuracy:.4f}")

        return seed, accuracy, None
    except Exception as e:
        print(f"Seed {seed}: Error – {e}")
        return seed, None, str(e)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global DATA_DIR
    
    parser = argparse.ArgumentParser(
        description="Fine-tune a correctness verifier with LoRA and evaluate best-of-N"
    )
    parser.add_argument("--train", type=str, required=True, help="train dataset name")
    parser.add_argument("--test", type=str, required=True, help="test dataset name")
    parser.add_argument("--model", type=str, default=None, help="model short name or HF id")
    parser.add_argument("--data-dir", type=str, default="/efs/cactts/data", help="base data directory")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of GPUs")
    parser.add_argument("--bs", type=int, default=1, help="per-device batch size for training")
    parser.add_argument("--grad-accum", type=int, default=16, help="gradient accumulation steps for training")
    parser.add_argument(
        "--sequential", action="store_true", help="run seeds sequentially"
    )
    args = parser.parse_args()
    
    DATA_DIR = args.data_dir

    seeds = [42, 52]
    turns = [1, 2, 3]

    model_mapping = {
        "gptoss": "openai/gpt-oss-20b",
        "ollmo7b": "allenai/OLMo-2-1124-7B-Instruct",
        "qwen1_7b": "Qwen/Qwen3-1.7B",
    }
    args.hf_model = model_mapping.get(args.model, args.model)

    print(f"Train: {args.train}")
    print(f"Test:  {args.test}")
    print(f"Model: {args.model}  →  {args.hf_model}")
    print(f"Seeds: {seeds}")
    print(f"GPUs:  {args.num_gpus}")
    print("=" * 70)

    tasks = []
    for idx, seed in enumerate(seeds):
        for turn in turns:
            gpu_id = idx % args.num_gpus
            tasks.append(
                (seed, gpu_id, args.train, args.test, args.model, args.hf_model, turn, args.bs, args.grad_accum)
            )

    results = []
    if args.sequential:
        for task in tasks:
            results.append(process_seed(task))
    else:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        print(f"\nParallel processing with {args.num_gpus} GPUs...")
        with mp.Pool(processes=min(len(tasks), args.num_gpus)) as pool:
            results = pool.map(process_seed, tasks)

    # ── report ──
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    accuracies = []
    for seed, accuracy, error in results:
        if error:
            print(f"Seed {seed}: Error – {error}")
        else:
            print(f"Seed {seed}: Accuracy = {accuracy:.4f}")
            accuracies.append(accuracy)

    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print(f"\n{'=' * 70}")
        print("FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Best-of-N Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Total experiments: {len(accuracies)}")

        train_name = Path(args.train).stem
        test_name = Path(args.test).stem
        output_file = f"results_sft_{train_name}_{test_name}.txt"

        with open(output_file, "w") as f:
            f.write("LoRA-based Correctness-Verifier Results\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"Train: {args.train}\n")
            f.write(f"Test:  {args.test}\n")
            f.write(f"Model: {args.hf_model}\n")
            f.write(f"Seeds: {seeds}\n\n")
            f.write(f"Best-of-N Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"\nIndividual results:\n")
            for seed, accuracy, error in results:
                if error:
                    f.write(f"  Seed {seed}: Error – {error}\n")
                else:
                    f.write(f"  Seed {seed}: {accuracy:.4f}\n")
            f.write(f"{'=' * 70}\n")

        print(f"\nResults saved to {output_file}")
    else:
        print("\nNo successful experiments completed.")


if __name__ == "__main__":
    main()
