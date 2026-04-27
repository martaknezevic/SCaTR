"""
Parse all_response_metrics.jsonl files and merge with original problem datasets.

Pipeline:
  1. Load per-turn response metrics from JSONL files.
  2. Retain a curated subset of columns.
  3. Load the corresponding problem dataset (HuggingFace or local JSONL).
  4. Join problem text onto the metrics dataframe via problem_id.
  5. Serialize the merged dataframe to parsed_data/<model>/<dataset>/turn{n}.pkl.
     Existing files are never overwritten.

Usage example
-------------
  python parse_responses.py \\
      --model Qwen/Qwen3-1.7B \\
      --dataset humaneval \\
"""

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

METRICS_COLUMNS = [
    "expected_answer",
    "extracted_answer",
    "is_correct",
    "mean_full",
    "mean_group_bottom_pct",
    "mean_group_highest",
    "mean_group_lowest",
    "mean_group_top_pct",
    "mean_tail",
    "problem_id",
    "response_text",
    "rollout_idx",
]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "aime": {
        "hf_dataset": "MathArena/aime_2025",
        "hf_config": None,
        "split": None,
        "local_file": None,
        "dir_path": "aime",
        "num_turns": 10,
        "id_columns": ["unique_id", "question_id", "task_id", "ID", "id", "problem_idx"],
        "problem_columns": ["problem", "question", "prompt", "Problem"],
        "id_prefix": "aime",
    },
    "aime24": {
        "hf_dataset": "Maxwell-Jia/AIME_2024",
        "hf_config": None,
        "split": "train",
        "local_file": None,
        "dir_path": "aime24",
        "num_turns": 10,
        "id_columns": ["ID"],
        "problem_columns": ["Problem"],
        "id_prefix": "aime24",
    },
    "gsm8k": {
        "hf_dataset": None,
        "hf_config": None,
        "split": None,
        "local_file": "./datasets/gsm8k_1000.jsonl",
        "dir_path": "gsm8k",
        "num_turns": 10,
        "id_columns": ["unique_id", "question_id", "task_id", "ID", "id"],
        "problem_columns": ["problem", "question", "prompt", "Problem"],
        "id_prefix": "gsm8k",
    },
    "humaneval": {
        "hf_dataset": None,
        "hf_config": None,
        "split": None,
        "local_file": "./datasets/humaneval.jsonl",
        "dir_path": "humaneval",
        "num_turns": 10,
        "id_columns": ["task_id", "question_id", "unique_id", "ID", "id"],
        "problem_columns": ["prompt", "problem", "question", "Problem"],
        "id_prefix": "HumanEval",
    },
    "kodcode": {
        "hf_dataset": None,
        "hf_config": None,
        "split": None,
        "local_file": "./datasets/kodcode_1000.jsonl",
        "dir_path": "kodcode",
        "num_turns": 10,
        "id_columns": ["question_id", "task_id", "unique_id", "ID", "id"],
        "problem_columns": ["prompt"],
        "id_prefix": "kodcode",
    },
    "math500": {
        "hf_dataset": "HuggingFaceH4/MATH-500",
        "hf_config": None,
        "split": "test",
        "local_file": None,
        "dir_path": "math500",
        "num_turns": 10,
        "id_columns": ["unique_id", "question_id", "task_id", "ID", "id", "problem_id"],
        "problem_columns": ["problem", "question", "prompt", "Problem"],
        "id_prefix": "math",
    },
    "bigcodebench_hard": {
        "hf_dataset": None,
        "hf_config": None,
        "split": None,
        "local_file": "./bigcodebench_hard.jsonl",
        "dir_path": "bigcodebench_hard",
        "num_turns": 10,
        "id_columns": ["task_id"],
        "problem_columns": ["complete_prompt"],
        "id_prefix": "kodcode",
    },
    "ot_code": {
        "hf_dataset": None,
        "hf_config": None,
        "split": None,
        "local_file": "./ot_code.jsonl",
        "dir_path": "ot_code",
        "num_turns": 10,
        "id_columns": ["id"],
        "problem_columns": ["prompt"],
        "id_prefix": "code",
    },
    "ot_math": {
        "hf_dataset": None,
        "hf_config": None,
        "split": None,
        "local_file": "./ot_math.jsonl",
        "dir_path": "ot_math",
        "num_turns": 10,
        "id_columns": ["id"],
        "problem_columns": ["question"],
        "id_prefix": "math",
    },
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_metrics_jsonl(filepath: Path) -> pd.DataFrame:
    """Load a response-metrics JSONL file and retain the canonical column set.

    Parameters
    ----------
    filepath:
        Path to an ``all_response_metrics.jsonl`` file.

    Returns
    -------
    pd.DataFrame
        DataFrame restricted to ``METRICS_COLUMNS`` (columns absent from the
        file are silently dropped from the selection with a warning).
    """
    print(f"  Loading {filepath}...")
    with open(filepath, "r") as fh:
        data = [json.loads(line) for line in fh]

    df = pd.DataFrame(data)

    available = [c for c in METRICS_COLUMNS if c in df.columns]
    missing = [c for c in METRICS_COLUMNS if c not in df.columns]
    if missing:
        print(f"    Warning: columns not found in metrics file and will be skipped: {missing}")

    df = df[available]
    print(f"  Loaded {len(df):,} rows, {len(available)} columns.")
    return df


def load_local_jsonl(filepath: str) -> pd.DataFrame:
    """Load a local JSONL file into a DataFrame.

    Parameters
    ----------
    filepath:
        Path to a ``.jsonl`` file on disk.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    print(f"  Loading local file: {path}")
    with open(path, "r") as fh:
        data = [json.loads(line) for line in fh]

    df = pd.DataFrame(data)
    print(f"  Loaded {len(df):,} rows.")
    return df


def load_hf_dataset_to_df(hf_dataset: str, hf_config: str | None, split: str | None) -> pd.DataFrame:
    """Download a HuggingFace dataset and return it as a single DataFrame.

    All available splits are concatenated when no explicit split is provided
    and the dataset exposes multiple splits.

    Parameters
    ----------
    hf_dataset:
        HuggingFace dataset identifier (e.g., ``"HuggingFaceH4/MATH-500"``).
    hf_config:
        Optional dataset configuration name.
    split:
        Optional split name. When ``None``, all splits are combined.

    Returns
    -------
    pd.DataFrame
    """
    print(f"  Loading from HuggingFace: {hf_dataset}")
    ds = load_dataset(hf_dataset, hf_config, split=split) if hf_config else load_dataset(hf_dataset, split=split)

    try:
        splits = list(ds.keys())
        print(f"  Available splits: {splits}")
        if len(splits) == 1:
            return ds[splits[0]].to_pandas()

        print("  Combining all splits...")
        frames = []
        for name in splits:
            frame = ds[name].to_pandas()
            print(f"    - {name}: {len(frame):,} rows")
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        print(f"  Combined: {len(combined):,} rows.")
        return combined

    except (AttributeError, TypeError):
        # A specific split was already resolved by load_dataset.
        df = ds.to_pandas()
        print(f"  Loaded {len(df):,} rows.")
        return df


def load_dataset_from_config(config: dict) -> pd.DataFrame:
    """Dispatch dataset loading to the appropriate backend.

    Parameters
    ----------
    config:
        Entry from ``DATASET_CONFIGS``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If neither ``local_file`` nor ``hf_dataset`` is specified.
    """
    if config.get("local_file"):
        return load_local_jsonl(config["local_file"])
    if config.get("hf_dataset"):
        return load_hf_dataset_to_df(config["hf_dataset"], config.get("hf_config"), config.get("split"))
    raise ValueError("Dataset config must specify either 'local_file' or 'hf_dataset'.")


def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Return the first column from ``candidates`` that exists in ``df``.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    candidates:
        Ordered list of candidate column names.
    label:
        Human-readable label used in the error message (e.g., ``"ID"``).

    Raises
    ------
    ValueError
        If none of the candidates are present.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find {label} column. Tried: {candidates}")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def build_problem_id_column(df: pd.DataFrame, dataset_name: str, id_prefix: str) -> pd.DataFrame:
    """Attach a normalised ``problem_id`` column to the original dataset.

    Dataset-specific ID conventions are handled here so that all downstream
    code can rely on a single, consistently named key column.

    Parameters
    ----------
    df:
        The raw original dataset DataFrame (modified in place and returned).
    dataset_name:
        Key from ``DATASET_CONFIGS``.
    id_prefix:
        Short string prefix used for index-based IDs (e.g., ``"math"``).

    Returns
    -------
    pd.DataFrame
        ``df`` with a ``problem_id`` column added or normalised.

    Raises
    ------
    ValueError
        If a required source column is absent.
    """
    name = dataset_name.lower()

    if name == "kodcode":
        if "question_id" in df.columns:
            df = df.rename(columns={"question_id": "problem_id"})
        elif "problem_id" not in df.columns:
            raise ValueError("Neither 'question_id' nor 'problem_id' found in KodCode dataset.")

    elif name == "humaneval":
        if "task_id" not in df.columns:
            raise ValueError("'task_id' column not found in HumanEval dataset.")
        df["problem_id"] = df["task_id"]

    elif name.startswith("bigcodebench"):
        if "task_id" not in df.columns:
            raise ValueError("'task_id' column not found in BigCodeBench dataset.")
        df["problem_id"] = df["task_id"]

    elif name == "aime24":
        if "ID" not in df.columns:
            raise ValueError("'ID' column not found in AIME-2024 dataset.")
        df["problem_id"] = df["ID"]

    else:
        # Fall back to sequential integer-based IDs.
        df["problem_id"] = [f"{id_prefix}_{i}" for i in range(len(df))]

    df["problem_id"] = df["problem_id"].astype(str)
    return df


def process_dataset(
    dataset_name: str,
    config: dict,
    base_dir: Path,
    output_dir: Path,
) -> None:
    """Parse metrics files for all turns of a dataset and save merged pickles.

    For each turn ``t`` in ``[1, config['num_turns']]``:
      - Reads ``<base_dir>/<dir_path>/turn{t}/all_response_metrics.jsonl``.
      - Merges the metrics with problem text from the original dataset.
      - Writes the result to ``<output_dir>/turn{t}.pkl`` unless the file
        already exists, in which case the turn is skipped.

    Parameters
    ----------
    dataset_name:
        Key from ``DATASET_CONFIGS``.
    config:
        Corresponding configuration dictionary.
    base_dir:
        Root data directory for the model under evaluation.
    output_dir:
        Destination directory for serialised output files.
    """
    print("\n" + "=" * 70)
    print(f"Processing {dataset_name.upper()} (turns 1–{config['num_turns']})")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load and prepare the original problem dataset.
    # ------------------------------------------------------------------
    print("Loading original dataset...")
    try:
        df_original = load_dataset_from_config(config)
    except Exception as exc:
        print(f"  Error loading dataset: {exc}")
        print(f"  Skipping {dataset_name}.")
        return

    print(f"  Columns available: {df_original.columns.tolist()}")
    df_original = df_original.reset_index(drop=True)

    try:
        df_original = build_problem_id_column(df_original, dataset_name, config["id_prefix"])
    except ValueError as exc:
        print(f"  Error constructing problem_id: {exc}")
        print(f"  Skipping {dataset_name}.")
        return

    try:
        problem_col = find_column(df_original, config["problem_columns"], "problem text")
    except ValueError as exc:
        print(f"  Error: {exc}")
        print(f"  Skipping {dataset_name}.")
        return

    print(f"  Problems: {len(df_original):,}  |  id_prefix: '{config['id_prefix']}'  |  problem column: '{problem_col}'")

    df_original = df_original.rename(columns={problem_col: "problem"})[["problem_id", "problem"]]

    # ------------------------------------------------------------------
    # Process each turn.
    # ------------------------------------------------------------------
    for turn in range(1, config["num_turns"] + 1):
        metrics_file = base_dir / config["dir_path"] / f"turn{turn}" / "all_response_metrics.jsonl"

        if not metrics_file.exists():
            print(f"\nTurn {turn}: metrics file not found, skipping.")
            continue

        output_file = output_dir / f"turn{turn}.pkl"
        if output_file.exists():
            print(f"\nTurn {turn}: output already exists at {output_file}, skipping.")
            continue

        print(f"\nTurn {turn}:")
        df_metrics = load_metrics_jsonl(metrics_file)

        if "problem_id" not in df_metrics.columns:
            print("    Warning: 'problem_id' column missing from metrics file.")
        else:
            df_metrics["problem_id"] = df_metrics["problem_id"].astype(str)

        # Merge problem text onto metrics rows.
        print("  Merging with original dataset...")
        df_merged = df_metrics.merge(df_original, on="problem_id", how="left")

        n_metrics_ids = df_metrics["problem_id"].nunique()
        n_original_ids = df_original["problem_id"].nunique()
        n_matching = len(set(df_metrics["problem_id"]) & set(df_original["problem_id"]))
        print(f"    Metrics problems: {n_metrics_ids}  |  Original problems: {n_original_ids}  |  Matched: {n_matching}")
        print(f"    Rows — metrics: {len(df_metrics):,}  merged: {len(df_merged):,}")

        n_missing = df_merged["problem"].isna().sum()
        if n_missing > 0:
            print(f"    Warning: {n_missing:,} rows have no matching problem text.")
        else:
            print(f"    All {len(df_merged):,} rows matched successfully.")

        df_merged.to_pickle(output_file)
        print(f"  Saved: {output_file}  ({output_file.stat().st_size / 1024**2:.2f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("PARSING RESPONSE METRICS")
    print("=" * 70)

    parser = argparse.ArgumentParser(
        description="Parse all_response_metrics.jsonl files and merge with original problem datasets."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name used to locate data under ./data/<model>/.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help=f"Dataset to process. Choices: {', '.join(DATASET_CONFIGS.keys())}",
    )
    args = parser.parse_args()

    base_dir = Path(f"tmp/scatr/data/{args.model}")
    output_dir = Path(f"./parsed_data/{args.model}/{args.dataset}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel:            {args.model}")
    print(f"Dataset:          {args.dataset}")
    print(f"Base directory:   {base_dir}")
    print(f"Output directory: {output_dir}")

    config = DATASET_CONFIGS[args.dataset]
    try:
        process_dataset(args.dataset, config, base_dir, output_dir)
    except Exception as exc:
        print(f"\nUnhandled error while processing {args.dataset}: {exc}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles written:")
    for f in sorted(output_dir.glob("*.pkl")):
        print(f"  {f.name:<30} ({f.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()