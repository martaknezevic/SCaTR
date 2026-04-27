#!/usr/bin/env python3
"""
Best-of-N evaluation using ReasonFlux-PRM-1.5B as a reward model.

Uses the same data loader and best-of-N selection logic as train_sft.py,
but replaces the fine-tuned yes/no verifier with the pre-trained
ReasonFlux-PRM-1.5B reward model (Gen-Verse/ReasonFlux-PRM-1.5B).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Reuse data loading from train_sft
from train_sft import load_data, cleanup_model

# ReasonFlux-PRM utilities — import directly to avoid collisions with other 'utils' packages
import importlib.util
_rm_utils_path = str(Path(__file__).resolve().parent.parent.parent / "ReasonFlux" / "ReasonFlux_PRM" / "utils" / "rm_utils.py")
_spec = importlib.util.spec_from_file_location("rm_utils", _rm_utils_path)
_rm_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rm_utils)
RewardModel = _rm_utils.RewardModel

PRM_MODEL_NAME = "Gen-Verse/ReasonFlux-PRM-1.5B"


def evaluate_best_of_n_prm(
    test_data: List[Dict],
    prm_model_name: str = PRM_MODEL_NAME,
    hf_token: str = None,
    batch_size: int = 16,
    max_length: int = 2048,
) -> Tuple[float, List[Dict], List[Dict], float, List[float]]:
    """
    Score every rollout with ReasonFlux-PRM-1.5B and pick the highest-reward
    rollout per problem (best-of-N).

    Args:
        test_data: list of dicts from load_data(), each with at least
                   'problem_id', 'problem', 'response_text', 'is_correct'.
        prm_model_name: HuggingFace repo id for the PRM.
        hf_token: HuggingFace auth token (needed to download reward_head.pt).
        batch_size: number of samples per forward pass.
        max_length: max token length for the tokenizer.

    Returns:
        accuracy: best-of-N accuracy across all problems.
        rollout_predictions: per-rollout reward scores.
        problem_results: per-problem best-of-N selection details.
        eval_time: total scoring wall time in seconds.
        infer_times: list of per-batch forward pass times in seconds.
    """
    test_data = test_data[:100]
    # ── load model & tokenizer ────────────────────────────────────────────
    print(f"Loading ReasonFlux-PRM from {prm_model_name}...")
    model = RewardModel.from_pretrained(prm_model_name, auth_token=hf_token)
    model.eval()
    device = model.base_model.device
    print(f"ReasonFlux-PRM loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        prm_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── build inputs: concatenate problem + response_text ─────────────────
    all_texts = []
    for item in test_data:
        response_ids = tokenizer(
            item["response_text"],
            add_special_tokens=False,
        )["input_ids"]
        # keep only the last max_length tokens of the response
        if len(response_ids) > max_length:
            response_ids = response_ids[-max_length:]
        truncated_response = tokenizer.decode(response_ids, skip_special_tokens=True)
        text = item["problem"] + "\n" + truncated_response
        all_texts.append(text)
    print(f"Built {len(all_texts)} inputs for PRM scoring.")

    # ── batched forward passes ────────────────────────────────────────────
    all_rewards: List[float] = []

    infer_times: List[float] = []
    eval_start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(all_texts), batch_size), desc="PRM scoring"):
            batch_texts = all_texts[i : i + batch_size]

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,   # response already truncated up front
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            start_infer = time.time()
            rewards = model(input_ids, attention_mask)  # shape: (B,)
            end_infer = time.time()
            infer_times.append(end_infer - start_infer)
            all_rewards.extend(rewards.cpu().tolist())

    eval_time = time.time() - eval_start_time

    print(f"Scored {len(all_rewards)} rollouts.")

    # ── per-rollout predictions ───────────────────────────────────────────
    rollout_predictions = []
    for idx, item in enumerate(test_data):
        rollout_predictions.append({
            "problem_id":  item["problem_id"],
            "rollout_idx": item.get("rollout_idx", idx),
            "reward":      all_rewards[idx],
            "is_correct":  item["is_correct"],
        })

    # ── best-of-N selection (highest reward per problem) ──────────────────
    problems: Dict[str, List[Dict]] = defaultdict(list)
    for idx, item in enumerate(test_data):
        problems[item["problem_id"]].append({
            "rollout":     item,
            "rollout_idx": item.get("rollout_idx", idx),
            "reward":      all_rewards[idx],
        })

    correct_predictions = 0
    total_problems = 0
    problem_results = []

    for problem_id, rollout_scores in problems.items():
        best = max(rollout_scores, key=lambda x: x["reward"])

        is_correct = best["rollout"]["is_correct"]
        if is_correct:
            correct_predictions += 1
        total_problems += 1

        problem_results.append({
            "problem_id":       problem_id,
            "best_rollout_idx": best["rollout_idx"],
            "best_reward":      best["reward"],
            "is_correct":       is_correct,
            "n_rollouts":       len(rollout_scores),
        })

    accuracy = correct_predictions / total_problems if total_problems > 0 else 0.0
    print(f"\nBest-of-N (PRM) evaluation: {correct_predictions}/{total_problems} = {accuracy:.2%}")

    # ── cleanup ───────────────────────────────────────────────────────────
    cleanup_model(model)
    del tokenizer

    return accuracy, rollout_predictions, problem_results, eval_time, infer_times


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best-of-N selection using ReasonFlux-PRM-1.5B"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset name (humaneval, kodcode, math500, aime)")
    parser.add_argument("--model", type=str, required=True,
                        help="generator model short name (used for data path)")
    parser.add_argument("--turn", type=int, default=1,
                        help="turn number")
    parser.add_argument("--bs", type=int, default=16,
                        help="batch size for PRM scoring")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="max token length for PRM input")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace auth token for downloading reward_head.pt")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="directory to save results (default: auto)")
    parser.add_argument("--inference_estimate", action="store_true",
                        help="estimate inference timing and log timing JSON")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
    print(f"Turn:    {args.turn}")
    print("=" * 70)

    # Load test data using the same loader as train_sft
    test_data = load_data(args.model, args.dataset, args.turn)
    print(f"Loaded {len(test_data)} rollouts")

    # Run PRM evaluation
    eval_start_time = time.time()
    accuracy, rollout_predictions, problem_results, eval_time, infer_times = evaluate_best_of_n_prm(
        test_data,
        hf_token=args.hf_token,
        batch_size=args.bs,
        max_length=args.max_length,
    )
    eval_end_time = time.time()

    # Save results
    if args.output_dir is None and args.inference_estimate:
        output_dir = Path(f"/tmp/scatr/models/inference_estimates_prm/{args.model}_{args.dataset}")
    elif args.output_dir is None:
        output_dir = Path(f"/tmp/scatr/models/baselines/prm_{args.model}_{args.dataset}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.inference_estimate:
        timing_path = output_dir / f"inference_timing_turn{args.turn}_{args.max_length}.json"
        with open(timing_path, "w") as f:
            json.dump({
                "prm_model": PRM_MODEL_NAME,
                "dataset": args.dataset,
                "gen_model": args.model,
                "turn": args.turn,
                "batch_size": args.bs,
                "max_length": args.max_length,
                "eval_bon_selection": eval_end_time - eval_start_time,
                "infer_times_seconds": infer_times,
                "eval_time_seconds": eval_time,
            }, f, indent=2)
        print(f"Saved inference timing → {timing_path}")
        return

    rollout_path = output_dir / f"prm_rollouts_turn{args.turn}_{args.max_length}.jsonl"
    with open(rollout_path, "w") as f:
        for row in rollout_predictions:
            f.write(json.dumps(row) + "\n")

    results_path = output_dir / f"prm_results_turn{args.turn}_{args.max_length}.json"
    with open(results_path, "w") as f:
        json.dump({
            "prm_model":   PRM_MODEL_NAME,
            "dataset":     args.dataset,
            "gen_model":   args.model,
            "turn":        args.turn,
            "accuracy":    accuracy,
            "n_correct":   int(accuracy * len(problem_results)),
            "n_problems":  len(problem_results),
            "problems":    problem_results,
        }, f, indent=2)

    print(f"Saved rollout scores → {rollout_path}")
    print(f"Saved results        → {results_path}")


if __name__ == "__main__":
    main()
