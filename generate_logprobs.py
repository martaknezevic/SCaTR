import argparse

"""
Reference-conditioned logprob evaluation.

For each rollout of each problem, select a reference rollout (correct or incorrect)
from the remaining rollouts, inject it into the prompt, then score the original
rollout's response text by computing per-token logprobs under the new context.
All confidence metrics (same as math_evaluator.py) are computed and saved.

Usage:
    python generate_logprobs.py \\
        --model <model_id_or_path> \\
        --dataset <dataset_name> \\
        --turn <turn_number> \\
        --correct \\
        --input_dir ./outputs/aime \\
        --output_dir ./outputs/aime_ref_correct
"""

import argparse
import asyncio
import json
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
from transformers import AutoTokenizer

from metrics import ConfidenceMetrics, aggregate_metrics
from storage import MetricsStorage, ChoiceStorage
from math_evaluator import LoaderFactory as MathLoaderFactory
from code_evaluator import LoaderFactory as CodeLoaderFactory

# ============================================================================
# Reference rollout selection
# ============================================================================

def select_reference_rollout(
    rollouts: List[Dict[str, Any]],
    current_rollout_idx: int,
    use_correct: bool,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """
    Pick a reference rollout from *rollouts*, excluding *current_rollout_idx*.

    Strategy:
    - use_correct=True  → prefer a correct rollout; fall back to incorrect if none.
    - use_correct=False → prefer an incorrect rollout; fall back to correct if none.

    Returns None when there are no other rollouts at all.
    """
    others = [r for r in rollouts if r["rollout_idx"] != current_rollout_idx]
    if not others:
        return None

    correct = [r for r in others if r.get("is_correct", False)]
    incorrect = [r for r in others if not r.get("is_correct", False)]

    if use_correct:
        pool = correct if correct else incorrect
    else:
        pool = incorrect if incorrect else correct

    return rng.choice(pool)


def build_messages(
    problem_text: str,
    reference_response: str,
    is_reference_correct: bool,
    current_response: str,
    system_prompt: str = "",
    model_id: str = "",
) -> List[Dict[str, str]]:
    """
    Build the message list that will be used as the prompt context.
    The initial rollout's response text is NOT included here — it is appended
    later as the completion to be scored.

    """
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    
    if 'gpt-oss' in model_id.lower() or 'gptoss' in model_id.lower():
        #parsed = parse_gptoss_response(current_response)
        parsed_ref = parse_gptoss_response(reference_response)       
        
        messages.append({
            "role": "user",
            "content": f"{problem_text}\n\nThis is a {is_reference_correct} reference response:\n{parsed_ref['thinking']}\n{parsed_ref['content']}\n", 
        })
        
        messages.append({
            "role": "assistant",
            'content': current_response
        })
    else:
        messages.append({
            "role": "user",
            "content": f"{problem_text}\n\nThis is a {is_reference_correct} reference response:\n{reference_response}\n",
        })
        
        messages.append({
            "role": "assistant",
            "content": current_response,
        })

    return messages

def parse_gptoss_response(response_text: str) -> Dict[str, str]:
    import re
    
    thinking = ""
    content  = ""
    
    # extract analysis/thinking
    analysis_match = re.search(
        r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
        response_text, re.DOTALL
    )
    if analysis_match:
        thinking = analysis_match.group(1).strip()
    
    # extract final content
    final_match = re.search(
        r'<\|start\|>assistant<\|channel\|>final(.*?)<\|return\|>',
        response_text, re.DOTALL
    )
    if final_match:
        content = final_match.group(1).strip()
    
    # remove any <|start|>assistant<|channel|>... artifacts from content
    content = re.sub(r'<\|start\|>assistant<\|channel\|>[^|]*\|>', '', content)
    content = re.sub(r'<\|constraint\|>output<\|message\|>', '', content)
    content = re.sub(r'<\|constraint\|>analysis<\|message\|>', '', content)
    content = re.sub(r'<\|[^\|]+\|>', '', content)  # remove all remaining special tokens
    content = content.strip()

    return {"thinking": thinking, "content": content}

async def generate_single_response_old(
    client: openai.AsyncOpenAI,
    config: Dict[str, Any],
) -> Any:
    """Generate a single response asynchronously."""
    response = await client.chat.completions.create(**config)
    return response.choices[0]


async def generate_single_response(
    client: openai.AsyncOpenAI,
    config: Dict[str, Any],
    tokenizer,
) -> Any:
    messages = config.pop("messages")
    
    prompt_messages = messages[:-1]
    assistant_content = messages[-1]["content"]
    assert messages[-1]["role"] == "assistant"
    
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + assistant_content

    response = await client.completions.create(
        model=config["model"],
        prompt=full_text,
        max_tokens=0,
        echo=True,
        logprobs=config.get("top_logprobs", 10),
    )

    choice = response.choices[0]
    all_tokens       = choice.logprobs.tokens
    all_logprobs     = choice.logprobs.token_logprobs
    all_top_logprobs = choice.logprobs.top_logprobs
    
    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    choice.prompt_len = prompt_len

    # attach directly here — not in caller
    choice.assistant_tokens       = all_tokens[prompt_len:]
    choice.assistant_logprobs     = all_logprobs[prompt_len:]
    choice.assistant_top_logprobs = all_top_logprobs[prompt_len:]

    return choice


def load_problem_text_map(dataset_name: str, dataset_source: str, split: str = "test") -> Dict[str, str]:
    """Load problem text mapping using the exact dataset loaders from math_evaluator.py / code_evaluator.py."""
    loader_type = dataset_name.lower()

    _CODE_LOADERS = {"humaneval", "kodcode", "generic_code"}
    _MATH_LOADERS = {"aime", "math", "gsm8k", "generic"}

    if loader_type in _CODE_LOADERS:
        loader = CodeLoaderFactory.create(loader_type)
    elif loader_type in _MATH_LOADERS:
        loader = MathLoaderFactory.create(loader_type)
    else:
        # unknown dataset – fall back to the math generic loader
        loader = MathLoaderFactory.create("generic")

    problems = loader.load(dataset_source, split=split)

    return {
        str(problem.problem_id): str(problem.problem_text)
        for problem in problems
        if getattr(problem, "problem_id", None) is not None and getattr(problem, "problem_text", None) is not None
    }


def augment_rollouts_with_problem_text(
    rollout_records: List[Dict[str, Any]],
    problem_text_map: Dict[str, str],
) -> int:
    """Populate rollout records with problem_text using a problem_id lookup."""
    augmented_count = 0
    for record in rollout_records:
        problem_id = str(record.get("problem_id", ""))
        problem_text = problem_text_map.get(problem_id)
        if problem_text and record.get("problem_text") != problem_text:
            record["problem_text"] = problem_text
            augmented_count += 1
    return augmented_count



# ============================================================================
# Saving helpers
# ============================================================================

def _save_pkl(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _build_metric_choice_payload(choice: Any) -> Dict[str, Any]:
    """Extract a picklable subset of the choice needed for metric computation."""
    return {
        "assistant_tokens": list(choice.assistant_tokens),
        "assistant_logprobs": list(choice.assistant_logprobs),
        "assistant_top_logprobs": [dict(top_lps or {}) for top_lps in choice.assistant_top_logprobs],
    }


def _compute_metrics_from_payload(
    choice_payload: Dict[str, Any],
    tail_n: int,
    group_size: int,
) -> Tuple[Dict[str, List[float]], Dict[str, Any], int]:
    """Compute rollout metrics in a worker process from a serializable payload."""
    metrics_calc = ConfidenceMetrics()
    choice = SimpleNamespace(
        assistant_tokens=choice_payload["assistant_tokens"],
        assistant_logprobs=choice_payload["assistant_logprobs"],
        assistant_top_logprobs=choice_payload["assistant_top_logprobs"],
    )
    metric_sequences = metrics_calc.compute_all_metrics(choice)
    aggregated = aggregate_metrics(
        metric_sequences,
        tail_n=tail_n,
        group_size=group_size,
    )
    total_tokens = len(metric_sequences.get("mean", []))
    return metric_sequences, aggregated, total_tokens


def _save_problem_outputs(
    problem_id: str,
    problem_rollout_objects: List[Dict[str, Any]],
    problem_records: List[Dict[str, Any]],
    choices_out_dir: str,
    metrics_file: str,
) -> None:
    """Persist per-problem outputs off the event loop."""
    if problem_rollout_objects:
        pkl_path = os.path.join(choices_out_dir, f"{problem_id}_choices.pkl")
        _save_pkl(problem_rollout_objects, pkl_path)

    if problem_records:
        MetricsStorage.save_all_metrics_batch(
            problem_records,
            metrics_file,
            use_compression=False,
        )


# ============================================================================
# Core evaluation loop
# ============================================================================

async def evaluate_all(
    client: openai.AsyncOpenAI,
    model_id: str,
    grouped: Dict[str, List[Dict[str, Any]]],
    choices_dir: str,
    output_dir: str,
    use_correct: bool,
    top_logprobs: int,
    tail_n: int,
    group_size: int,
    system_prompt: str,
    max_concurrent: int,
    batch_size: int,
    rng: random.Random,
) -> None:
    """
    For every problem and every rollout:
      1. Select a reference rollout.
      2. Build prompt messages (user-defined).
      3. Score the initial rollout's response under the new context.
      4. Compute confidence metrics.
            5. Save per-problem pkl (all rollouts) + append to JSONL.
    """
    choices_out_dir = os.path.join(output_dir, "choices")
    metrics_file = os.path.join(output_dir, "ref_response_metrics.jsonl")
    os.makedirs(choices_out_dir, exist_ok=True)

    problem_ids = list(grouped.keys())
    print(f"Scoring {len(problem_ids)} problems in batches of {batch_size} …")
    loop = asyncio.get_running_loop()
    metric_workers = os.cpu_count() or 4
    save_workers = max(1, min(8, metric_workers))
    from transformers import AutoTokenizer
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    async def process_rollout(
        problem_id: str,
        rollout: Dict[str, Any],
        all_rollouts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        rollout_idx      = rollout["rollout_idx"]
        response_text    = rollout.get("response_text", "")
        is_correct_cur   = rollout.get("is_correct", False)

        # 1. Select reference
        ref = select_reference_rollout(all_rollouts, rollout_idx, use_correct, rng)
        if ref is None:
            print(f"  [{problem_id}] rollout {rollout_idx}: no reference available, skipping.")
            return None

        ref_response_text = ref.get("response_text", "")
        ref_is_correct    = ref.get("is_correct", False)
        
        
        # 2. Build prompt messages
        messages = build_messages(
            problem_text=rollout.get("problem_text"),
            reference_response=ref_response_text,
            is_reference_correct=ref_is_correct,
            system_prompt=system_prompt,
            current_response=response_text,
            model_id=model_id,
        )

        
        #token_ids = tokenizer.apply_chat_template(
        #    messages,
        #    tokenize=True,
        #    add_generation_prompt=True
        #)
        #print(f"Token count: {len(token_ids)}")

        threshold = 131072 if 'gpt-oss' in model_id.lower() or 'gptoss' in model_id.lower() else 40960
        truncate_threshold = 32768 if 'gpt-oss' in model_id.lower() or 'gptoss' in model_id.lower() else 16384
        
        if not ('gpt-oss' in model_id.lower() or 'gptoss' in model_id.lower()):
            if len(token_ids) > threshold:
                # truncate messages
                for message in messages:
                    if message['role'] in ['user', 'assistant']:
                        tokens = tokenizer.encode(message['content'], add_special_tokens=False)
                        if len(tokens) > truncate_threshold:
                            message['content'] = tokenizer.decode(tokens[-truncate_threshold:], skip_special_tokens=True)
                            
                        
        # 3. Create generation config
        gen_config = {
            "model": model_id,
            "messages": messages,
            "original_response": response_text,
            'problem_id': problem_id,
            'rollout_idx': rollout_idx,
        }

        choice = await generate_single_response(client, gen_config, tokenizer)
        
        if not getattr(choice, "logprobs", None) or not getattr(choice.logprobs, "tokens", None):
            raise ValueError(
                f"No assistant-token logprobs returned for problem {problem_id}, rollout {rollout_idx}."
            )

        # 4. Compute metrics in a process pool; this is CPU-bound.
        choice_payload = _build_metric_choice_payload(choice)
        metric_sequences, aggregated, total_tokens = await loop.run_in_executor(
            metric_executor,
            _compute_metrics_from_payload,
            choice_payload,
            tail_n,
            group_size,
        )

        # 5a. Build JSONL record
        record: Dict[str, Any] = {
            "problem_id":       problem_id,
            "rollout_idx":      rollout_idx,
            "ref_rollout_idx":  ref["rollout_idx"],
            "ref_is_correct":   ref_is_correct,
            "is_correct":       is_correct_cur,
            "response_text":    response_text,
            "total_tokens":     total_tokens,
            **aggregated,
        }

        # 5b. Build per-rollout object for per-problem pickle
        rollout_obj: Dict[str, Any] = {
            "problem_id":        problem_id,
            "rollout_idx":       rollout_idx,
            "ref_rollout_idx":   ref["rollout_idx"],
            "ref_is_correct":    ref_is_correct,
            "is_correct":        is_correct_cur,
            "response_text":     response_text,
            "ref_response_text": ref_response_text,
            "choice":            choice,
            "metric_sequences":  metric_sequences,
            "aggregated_metrics": aggregated,
            "total_tokens":      record["total_tokens"],
        }

        return {
            "record": record,
            "rollout_obj": rollout_obj,
        }

    async def process_problem(problem_id: str) -> int:
        rollouts = grouped[problem_id]
        
        # process all rollouts concurrently instead of sequentially
        results = await asyncio.gather(
            *[process_rollout(problem_id, rollout, rollouts) for rollout in rollouts],
            return_exceptions=True  
        )
        
        problem_records = []
        problem_rollout_objects = []
        for result in results:
            if isinstance(result, Exception):
                print(f"  [{problem_id}] rollout error: {result}")
                continue
            if result is not None:
                problem_records.append(result["record"])
                problem_rollout_objects.append(result["rollout_obj"])

        if problem_rollout_objects or problem_records:
            loop.run_in_executor(
                save_executor,
                _save_problem_outputs,
                problem_id,
                problem_rollout_objects,
                problem_records,
                choices_out_dir,
                metrics_file,
            )
        return len(problem_records)

    async def process_problem_with_semaphore(
        problem_id: str,
        semaphore: asyncio.Semaphore,
    ) -> int:
        async with semaphore:
            return await process_problem(problem_id)

    with ProcessPoolExecutor(max_workers=metric_workers) as metric_executor, ThreadPoolExecutor(max_workers=save_workers) as save_executor:
        total_ok = 0
        total_err = 0
        total_batches = (len(problem_ids) + batch_size - 1) // batch_size

        for batch_start in range(0, len(problem_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(problem_ids))
            current_batch = problem_ids[batch_start:batch_end]
            current_batch_num = batch_start // batch_size + 1

            print(
                f"\nProcessing batch {current_batch_num}/{total_batches} ({len(current_batch)} problems, "
                f"{metric_workers} metric workers, {save_workers} save workers)"
            )
            semaphore = asyncio.Semaphore(max_concurrent)
            batch_tasks = [
                process_problem_with_semaphore(problem_id, semaphore)
                for problem_id in current_batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            batch_ok = 0
            batch_err = 0
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    batch_err += 1
                    print(f"  Error in problem {current_batch[idx]}: {result}")
                    continue
                batch_ok += result

            total_ok += batch_ok
            total_err += batch_err
            print(f"  Batch {current_batch_num} wrote {batch_ok} rollout records ({batch_err} problem-level errors).")

    n_ok = total_ok
    n_err = total_err
    print(f"\nDone. Scored {n_ok} rollouts, {n_err} errors.")
    print(f"Metrics : {metrics_file}")
    print(f"Pkl dir : {choices_out_dir}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score existing rollouts under a reference-conditioned prompt."
    )
    p.add_argument("--model_name",   default=None,
                   help="Model name.")
    p.add_argument("--dataset", required=True,
                   help="Dataset name (used to locate input_dir if not set).")
    p.add_argument("--turn",    type=int, default=1,
                   help="Evaluation turn number (for bookkeeping / folder naming).")
    p.add_argument("--correct", action="store_true",
                   help="If set, prefer correct rollouts as reference; "
                        "otherwise prefer incorrect ones.")

    p.add_argument("--input_dir",  default=None,
                   help="Folder containing all_response_metrics.jsonl and choices/. ")
    p.add_argument("--output_dir", default=None,
                   help="Where to write results. ")

    p.add_argument("--top_logprobs",  type=int, default=10)
    p.add_argument("--tail_n",        type=int, default=2048)
    p.add_argument("--group_size",    type=int, default=1024)
    p.add_argument("--max_concurrent",type=int, default=10,
                   help="Max number of concurrent scoring API calls.")
    p.add_argument("--batch_size", type=int, default=10,
                   help="Number of problems per batch.")
    p.add_argument("--seed",          type=int, default=42)

    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--api_key",  default="token-abc")
    p.add_argument("--timeout",  type=int, default=3600)
    p.add_argument("--dataset_source", default=None,
                   help="HF dataset name or local JSONL/JSON file for problem text lookup.")
    p.add_argument("--dataset_split", default="test",
                   help="Split to use when loading a Hugging Face dataset source.")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)
    
    # ── OpenAI client ────────────────────────────────────────────────────────
    client = openai.AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    # Auto-detect model id if needed (the served model name may differ)
    models = await client.models.list()
    served_model_id = models.data[0].id
    print(f"Served model: {served_model_id}")
    
    model_mapping = {
        "Qwen/Qwen3-1.7B": "qwen1_7b",
        "openai/gpt-oss-20b": "gptoss"
    }
    
    system_prompts = {
        'aime': """You are solving an AIME (American Invitational Mathematics Examination) problem. 
        Show your work step by step and provide your final numerical answer clearly. 
        AIME answers are integers from 0 to 999.
        """,
        'humaneval': """You are a Python coding AI agent. When given a Python function signature and docstring, you must provide a complete Python solution following this exact format:

    Reasoning Phase: Use <think>...</think> tags to contain your complete thought process

    Break down the problem requirements step by step
    Identify key constraints, edge cases, and potential pitfalls
    Plan your algorithm and data structures
    Walk through examples to validate your approach
    Explain your logic thoroughly as if working on scratch paper

    Implementation Phase: Use <output>...</output> tags to contain your final code

    Include the complete function with the original signature
    Ensure your code directly implements the approach from your thinking
    Write clean, readable code with appropriate comments if needed

    Your solution must be complete, correct, and handle all specified requirements.""",
        'kodcode': """You are a Python coding AI agent. When given a Python function signature and docstring, you must provide a complete Python solution following this exact format:

    Reasoning Phase: Use <think>...</think> tags to contain your complete thought process

    Break down the problem requirements step by step
    Identify key constraints, edge cases, and potential pitfalls
    Plan your algorithm and data structures
    Walk through examples to validate your approach
    Explain your logic thoroughly as if working on scratch paper

    Implementation Phase: Use <output>...</output> tags to contain your final code

    Include the complete function with the original signature
    Ensure your code directly implements the approach from your thinking
    Write clean, readable code with appropriate comments if needed

    Your solution must be complete, correct, and handle all specified requirements."""
    }
    
    model_name = model_mapping.get(served_model_id)
    if args.model_name:
        assert args.model_name == model_name, f"Provided model_name '{args.model_name}' does not match detected model '{model_name}'"
    else:
        args.model_name = model_name
        print(f"Detected model_name: {args.model_name}")
        
    args.system_prompt = system_prompts.get(args.dataset, "")
    if not args.system_prompt:
        raise ValueError(f"No system prompt defined for dataset '{args.dataset}'.")

    # ── Resolve directories ──────────────────────────────────────────────────
    ref_kind   = "correct" if args.correct else "incorrect"
    input_dir  = args.input_dir  or os.path.join("/efs/cactts/data", args.model_name, args.dataset, f"turn{args.turn}")
    output_dir = args.output_dir or os.path.join(
        "/efs/cactts/ref_data", f"{args.model_name}", f"{args.dataset}", f"turn{args.turn}", f"ref_{ref_kind}"
    )

    metrics_file = os.path.join(input_dir, "all_response_metrics.jsonl")
    choices_dir  = os.path.join(input_dir, "choices")

    if not os.path.isfile(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    dataset_to_problems = {
        'aime':      "MathArena/aime_2025",
        'humaneval': "datasets/humaneval.jsonl",
        'kodcode':   "datasets/kodcode_1000.jsonl",
    }
    dataset_source = args.dataset_source or dataset_to_problems.get(args.dataset)

    # ── Load and group metrics ───────────────────────────────────────────────
    print(f"Loading {metrics_file} …")
    rollout_records: List[Dict[str, Any]] = []
    with open(metrics_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rollout_records.append(json.loads(line))

    if dataset_source:
        print(f"Loading problem text mapping from {dataset_source} …")
        problem_text_map = load_problem_text_map(args.dataset, dataset_source, split=args.dataset_split)
        augmented_count = augment_rollouts_with_problem_text(rollout_records, problem_text_map)
        print(f"  Augmented {augmented_count} rollout records with problem_text.")
    else:
        print("No dataset source configured for problem_text augmentation.")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in rollout_records:
        grouped[rec["problem_id"]].append(rec)

    print(f"  {len(rollout_records)} rollout records across {len(grouped)} problems.")

    # ── Enrich records with response_text from choice pkl files (if missing) ─
    # all_response_metrics.jsonl already contains response_text; but if a run
    # was made without it, we fall back to loading from the pkl choice files.
    missing_text = any("response_text" not in r or not r["response_text"] for r in rollout_records)
    if missing_text and os.path.isdir(choices_dir):
        print("response_text missing in some records — loading from choice pkl files …")
        for problem_id, rollouts in grouped.items():
            pkl_path = os.path.join(choices_dir, f"{problem_id}_choices.pkl")
            if not os.path.isfile(pkl_path):
                continue
            choices = ChoiceStorage.load_choices(pkl_path)
            choice_map = {getattr(c, "rollout_idx", i): c for i, c in enumerate(choices)}
            for rec in rollouts:
                if not rec.get("response_text"):
                    ch = choice_map.get(rec["rollout_idx"])
                    if ch is not None and hasattr(ch, "logprobs") and ch.logprobs and ch.logprobs.content:
                        rec["response_text"] = "".join(
                            t.token for t in ch.logprobs.content if hasattr(t, "token")
                        )

    

    # ── Run evaluation ───────────────────────────────────────────────────────
    print(f"\nReference selection: {'correct' if args.correct else 'incorrect'}")
    print(f"Output dir        : {output_dir}\n")

    await evaluate_all(
        client=client,
        model_id=served_model_id,
        grouped=grouped,
        choices_dir=choices_dir,
        output_dir=output_dir,
        use_correct=args.correct,
        top_logprobs=args.top_logprobs,
        tail_n=args.tail_n,
        group_size=args.group_size,
        system_prompt=args.system_prompt,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        rng=rng,
    )


if __name__ == "__main__":
    asyncio.run(main())