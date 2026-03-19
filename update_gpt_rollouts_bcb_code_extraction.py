#!/usr/bin/env python3
"""
Extracts code from response_text in all_response_metrics.jsonl files.
Adds 'extracted_code' field to each record using HybridExtractor (markdown + output tags).

Usage:
    python extract_code.py
"""

import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from regrade_responses import MarkdownCodeExtractor, OutputTagExtractor

BASE_DIR = "/efs/cactts/data"
MODELS   = ["gptoss"]
DATASETS = ["bigcodebench_hard"]
TURNS    = [1, 2, 3]


def extract_record(record: dict) -> dict:
    """Extract code from a single record using markdown extractor, fallback to output tags."""
    text        = record.get("response_text", "")
    entry_point = record.get("entry_point")
    
    message = re.search(
        r'<\|start\|>assistant<\|channel\|>.*?<\|message\|>(.*?)<\|return\|>',
        text,
        re.DOTALL
    )
    text = message.group(1) if message else text
    
    if '<|start|>assistant<|channel|>output>' in text:
        text = text.split('<|start|>assistant<|channel|>output>')[-1]
    if "<|return|>" in text:
        text = text.replace("<|return|>", "")
    if '<|message|>' in text:
        text = text.split('<|message|>')[-1]
    if '<code>' in text:
        text = text.split('<code>')[-1]
        if '</code>' in text:
            text = text.split('</code>')[0]
    if '<output code>' in text:
        text = text.split('<output code>')[-1]
        if '</output code>' in text:
            text = text.split('</output code>')[0]
    if '<output>' in text:
        text = text.split('<output>')[-1]
        if '</output>' in text:
            text = text.split('</output>')[0]
    if '<output codeblock=\"python\">' in text:
        text = text.split('<output codeblock=\"python\">')[-1]
        if '</output codeblock>' in text:
            text = text.split('</output codeblock>')[0]
        if '</output>' in text:
            text = text.split('</output>')[0]
    if '<output code=\"python\">' in text:
        text = text.split('<output code=\"python\">')[-1]
        if '</output code>' in text:
            text = text.split('</output code>')[0]
        if '</output>' in text:
            text = text.split('</output>')[0]

    code = MarkdownCodeExtractor().extract(text, entry_point)
    if not code:
        code = OutputTagExtractor().extract(text, entry_point)

    record["extracted_code"] = code or ""
    return record


def process_one(model: str, dataset: str, turn: int) -> str:
    metrics_file = os.path.join(BASE_DIR, model, dataset, f"turn{turn}", "all_response_metrics.jsonl")
    tag = f"{model}/{dataset}/turn{turn}"

    if not os.path.exists(metrics_file):
        return f"SKIP {tag} — metrics file not found"

    with open(metrics_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        return f"SKIP {tag} — empty file"

    n_workers = min(len(records), os.cpu_count() or 4)
    results = [None] * len(records)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(extract_record, rec): i for i, rec in enumerate(records)}
        with tqdm(total=len(records), desc=tag, unit="rec") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = records[idx]  # keep original on error
                    print(f"  [{tag}] ERROR on record {idx}: {e}")
                pbar.update(1)

    shutil.copy(metrics_file, metrics_file + ".bak")
    with open(metrics_file, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    return f"OK {tag} — extracted code for {len(records)} records"


def main():
    tasks = [
        (model, dataset, turn)
        for model   in MODELS
        for dataset in DATASETS
        for turn    in TURNS
    ]

    print(f"Processing {len(tasks)} tasks...\n")
    for model, dataset, turn in tasks:
        try:
            print(process_one(model, dataset, turn))
        except Exception as e:
            print(f"ERROR {model}/{dataset}/turn{turn}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()