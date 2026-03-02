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

DATA_DIR = "/efs/cactts/data"


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
    "Is this answer correct?"
)


def build_tokenized_dataset(
    data: List[Dict], tokenizer, max_seq_length: int = 2048,
) -> HFDataset:
    """
    Build a HuggingFace Dataset with ``input_ids``, ``attention_mask``, and
    ``labels`` columns.  Labels are -100 for all prompt tokens so loss is only
    computed on the assistant's "true"/"false" response.
    """
    all_input_ids, all_attn, all_labels = [], [], []

    for item in data:
        label_text = "true" if item["is_correct"] else "false"
        user_content = USER_TEMPLATE.format(
            problem=item["problem"],
            response_text=item["response_text"],
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
            tokenize=False, add_generation_prompt=False,
        )

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate to max_seq_length
        full_ids = full_ids[:max_seq_length]
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
    gpu_id: int,
    max_seq_length: int = 2048,
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

    # Log token info
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    print(f"'true'  token ids: {true_ids}")
    print(f"'false' token ids: {false_ids}")

    # ── dataset (pre-tokenized with label masking) ──
    dataset = build_tokenized_dataset(train_data, tokenizer, max_seq_length)
    val_dataset = build_tokenized_dataset(val_data, tokenizer, max_seq_length)
    print(f"Training examples: {len(dataset)}, Validation examples: {len(val_dataset)}")
    # Show a decoded sample so we can verify formatting
    sample_ids = dataset[0]["input_ids"]
    sample_labels = dataset[0]["labels"]
    print(f"Sample prompt + response:\n{tokenizer.decode(sample_ids)[:500]}")
    n_train_tokens = sum(1 for l in sample_labels if l != -100)
    print(f"Tokens with loss in first example: {n_train_tokens}")

    collator = CompletionOnlyCollator(tokenizer=tokenizer)

    # ── compute expected steps for user info ──
    num_examples = len(dataset)
    batch_size = 8
    grad_accum = 4
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
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
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
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.train()
    print(f"\nTraining complete. Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    sys.stdout.flush()
    return output_dir


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_best_of_n(
    model_dir: str, test_data: List[Dict], gpu_id: int,
) -> float:
    """
    Best-of-N selection: for each problem pick the rollout whose next-token
    probability of "true" is highest.  Falls back to lowest P("false") when
    every rollout predicts "false".
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ensure_chat_template(tokenizer)

    # Load base model and LoRA adapters
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    true_token_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("false", add_special_tokens=False)[0]

    # Group rollouts by problem
    problems = defaultdict(list)
    for item in test_data:
        problems[item["problem_id"]].append(item)

    correct_predictions = 0
    total_problems = 0

    with torch.no_grad():
        for problem_id, rollouts in tqdm(problems.items(), desc="Evaluating"):
            rollout_scores = []
            for rollout in rollouts:
                # Build the user-assistant prompt (no assistant content yet)
                messages = [
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            problem=rollout["problem"],
                            response_text=rollout["response_text"],
                        ),
                    },
                ]
                # add_generation_prompt=True appends the assistant header
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                logits = model(**inputs).logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

                true_prob = probs[true_token_id].item()
                false_prob = probs[false_token_id].item()
                predicted = true_token_id if true_prob > false_prob else false_token_id

                rollout_scores.append(
                    {
                        "rollout": rollout,
                        "true_prob": true_prob,
                        "false_prob": false_prob,
                        "predicted_token": predicted,
                    }
                )

            # Pick best rollout
            true_preds = [
                r for r in rollout_scores if r["predicted_token"] == true_token_id
            ]
            if true_preds:
                best = max(true_preds, key=lambda x: x["true_prob"])
            else:
                best = min(rollout_scores, key=lambda x: x["false_prob"])

            if best["rollout"]["is_correct"]:
                correct_predictions += 1
            total_problems += 1

    return correct_predictions / total_problems if total_problems > 0 else 0.0


# ── per-seed entry-point ─────────────────────────────────────────────────────

def process_seed(args_tuple):
    """Process a single (seed, turn) experiment."""
    seed, gpu_id, train_ds, test_ds, model_name, hf_model, turn = args_tuple

    # Set CUDA device *before* any torch / CUDA call so the correct GPU is
    # used from the very first allocation (critical in multiprocessing workers).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        print(f"\nSeed {seed}, Turn {turn} (GPU {gpu_id}): Starting...")

        train_data = load_data(model_name, train_ds, turn)
        test_data = load_data(model_name, test_ds, turn)

        # Split training data into train/val by problem_id
        train_split, val_split = split_train_val(train_data, val_ratio=0.2, random_seed=seed)
        print(
            f"Seed {seed}: {len(train_split)} train / {len(val_split)} val / {len(test_data)} test samples"
        )

        output_dir = f"./sft_output/seed_{seed}_turn{turn}"

        print(f"Seed {seed}: Training...")
        model_dir = train_model(train_split, val_split, hf_model, output_dir, seed, gpu_id)

        print(f"Seed {seed}: Evaluating...")
        accuracy = evaluate_best_of_n(model_dir, test_data, gpu_id)
        print(f"Seed {seed}: Accuracy = {accuracy:.4f}")

        return seed, accuracy, None
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
    parser.add_argument("--num-gpus", type=int, default=3, help="number of GPUs")
    parser.add_argument(
        "--sequential", action="store_true", help="run seeds sequentially"
    )
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
                (seed, gpu_id, args.train, args.test, args.model, args.hf_model, turn)
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
