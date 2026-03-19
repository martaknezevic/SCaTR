#!/usr/bin/env python3
import json
import pickle
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_full_rollout_text(rollout_obj):
    full_text = ""
    for token in rollout_obj.logprobs.content:
        full_text += token.token
    return full_text

def process_one(model: str, dataset: str, turn: int, base_dir: str) -> str:
    """Process a single (model, dataset, turn) combination — runs in subprocess."""
    metrics_file = os.path.join(base_dir, model, dataset, f"turn{turn}", "all_response_metrics.jsonl")
    choices_dir  = os.path.join(base_dir, model, dataset, f"turn{turn}", "choices")
    tag          = f"{model}/{dataset}/turn{turn}"

    if not os.path.exists(metrics_file):
        return f"SKIP {tag} — metrics file not found"

    with open(metrics_file, 'r') as f:
        all_records = [json.loads(line) for line in f]

    updated = 0
    for record in all_records:
        if record.get('response_text') == "":
            problem_id  = record['problem_id']
            rollout_idx = record['rollout_idx']
            choices_path = os.path.join(
                choices_dir,
                f"{problem_id.replace('/', '_')}_choices.pkl"
            )
            try:
                with open(choices_path, 'rb') as f:
                    choices = pickle.load(f)
                record['response_text'] = get_full_rollout_text(choices[rollout_idx])
                updated += 1
            except FileNotFoundError:
                print(f"  [{tag}] WARNING: choices file not found for {problem_id}")
            except (IndexError, KeyError) as e:
                print(f"  [{tag}] WARNING: rollout {rollout_idx} error for {problem_id}: {e}")
            except Exception as e:
                print(f"  [{tag}] WARNING: {problem_id}: {e}")

    if updated > 0:
        shutil.copy(metrics_file, metrics_file + '.bak')
        with open(metrics_file, 'w') as f:
            for record in all_records:
                f.write(json.dumps(record) + '\n')

    return f"OK {tag} — updated {updated} records"


def main():
    models   = ["gptoss"]
    datasets = ["bigcodebench_hard"] #"humaneval", "kodcode"]
    turns    = [1, 2, 3]
    base_dir = "/efs/cactts/data"

    tasks = [
        (model, dataset, turn)
        for model   in models
        for dataset in datasets
        for turn    in turns
    ]

    max_workers = min(len(tasks), os.cpu_count() or 4)
    print(f"Processing {len(tasks)} tasks with {max_workers} workers...\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, model, dataset, turn, base_dir): (model, dataset, turn)
            for model, dataset, turn in tasks
        }
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                model, dataset, turn = futures[future]
                print(f"ERROR {model}/{dataset}/turn{turn}: {e}")


if __name__ == "__main__":
    main()