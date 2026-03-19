#!/usr/bin/env python3
import json
import pickle
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
from tqdm import tqdm

def get_full_rollout_text(rollout_obj) -> str:
    full_text = ""
    for token in rollout_obj.logprobs.content:
        full_text += token.token
    return full_text

def process_one_pickle(args):
    """Process a single pickle file and return (problem_id, {rollout_idx: response_text})."""
    choices_path, problem_id = args
    result = {}
    try:
        with open(choices_path, 'rb') as f:
            choices = pickle.load(f)
        for rollout_idx, choice in enumerate(choices):
            try:
                result[rollout_idx] = get_full_rollout_text(choice)
            except Exception as e:
                print(f"  WARNING: [{problem_id}] rollout {rollout_idx}: {e}")
    except Exception as e:
        print(f"  WARNING: could not load {choices_path}: {e}")
    return problem_id, result

def process_one(model: str, dataset: str, turn: int, base_dir: str) -> str:
    metrics_file = os.path.join(base_dir, model, dataset, f"turn{turn}", "all_response_metrics.jsonl")
    choices_dir  = os.path.join(base_dir, model, dataset, f"turn{turn}", "choices")
    tag          = f"{model}/{dataset}/turn{turn}"

    if not os.path.exists(metrics_file):
        return f"SKIP {tag} — metrics file not found"
    if not os.path.exists(choices_dir):
        return f"SKIP {tag} — choices dir not found"

    with open(metrics_file, 'r') as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    problem_ids = set(record['problem_id'] for record in all_records)

    pickle_tasks = []
    for problem_id in problem_ids:
        choices_path = os.path.join(
            choices_dir,
            f"{problem_id.replace('/', '_')}_choices.pkl"
        )
        if os.path.exists(choices_path):
            pickle_tasks.append((choices_path, problem_id))
        else:
            print(f"  [{tag}] WARNING: choices file not found for {problem_id}")

    if not pickle_tasks:
        return f"SKIP {tag} — no pickle files found"

    # load all pickles in parallel with tqdm
    n_workers = min(len(pickle_tasks), os.cpu_count() or 4)
    response_map: Dict[str, Dict[int, str]] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_one_pickle, task): task for task in pickle_tasks}
        with tqdm(total=len(pickle_tasks), desc=f"{tag}", unit="pkl") as pbar:
            for future in as_completed(futures):
                try:
                    problem_id, rollout_texts = future.result()
                    response_map[problem_id] = rollout_texts
                except Exception as e:
                    _, problem_id = futures[future]
                    print(f"  [{tag}] ERROR loading {problem_id}: {e}")
                pbar.update(1)

    # update all records
    updated = 0
    for record in tqdm(all_records, desc=f"{tag} updating records", unit="rec"):
        problem_id  = record['problem_id']
        rollout_idx = record['rollout_idx']
        text = response_map.get(problem_id, {}).get(rollout_idx)
        if text is not None:
            record['response_text'] = text
            updated += 1

    if updated > 0:
        shutil.copy(metrics_file, metrics_file + '.bak3')
        with open(metrics_file, 'w') as f:
            for record in tqdm(all_records, desc=f"{tag} writing", unit="rec"):
                f.write(json.dumps(record) + '\n')

    return f"OK {tag} — updated {updated}/{len(all_records)} records"


def main():
    models   = ["gptoss"]
    datasets = ['bigcodebench_hard'] #["humaneval", "kodcode"]
    turns    = [1, 2, 3]
    base_dir = "/efs/cactts/data"

    tasks = [
        (model, dataset, turn)
        for model   in models
        for dataset in datasets
        for turn    in turns
    ]

    print(f"Processing {len(tasks)} (model, dataset, turn) tasks sequentially...\n")

    for model, dataset, turn in tqdm(tasks, desc="Overall progress", unit="task"):
        try:
            result = process_one(model, dataset, turn, base_dir)
            print(result)
        except Exception as e:
            print(f"ERROR {model}/{dataset}/turn{turn}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()