# SCaTR

This repository contains code for running code and math generation/evaluation workflows, along with dataset configs and baseline training scripts.

## Installation

Install the Python dependencies from the provided `requirements.txt` files. The main project dependencies are in `requirements.txt`.

```bash
pip install -r requirements.txt
```

If you want to run the PRM baseline scripts, also install the baseline requirements:

```bash
pip install -r baselines/requirements_prm.txt
```

## Start the Code Executor

Before running code generation or evaluation jobs, launch the code executor service:

```bash
gunicorn code_executor_api:app \
    -k uvicorn.workers.UvicornWorker \
    -w 8 \
    -b 0.0.0.0:8001 \
    --timeout 10 \
    --keep-alive 5 \
    --log-level info
```

## Start the Model Server

For data generation, serve the model with vLLM:

```bash
vllm serve model-name --async-scheduling --enable-prefix-caching --data-parallel-size 4
```

The generation scripts in this repository expect a vLLM-compatible OpenAI API endpoint. The provided scripts use the default `http://localhost:8000/v1` base URL.

## Generate Data

Use the scripts in `scripts/generation/` to generate data.

Provided scripts for Qwen 1.7B:

- `scripts/generation/run_code_qwen1_7b.sh`
- `scripts/generation/run_math_qwen1_7b.sh`

These scripts already include the temperatures and other parameters used for the Qwen 1.7B setup.

For other models, use the same scripts or adapt them with the model-specific parameters you want to run. In particular, set the temperature explicitly when needed. For example, use `temperature=1` for GPT-OSS and OLMo-style runs if that is the intended setup.

## Configs and Datasets

Dataset generation and evaluation settings are defined in `configs/`.

Available configs include:

- `configs/config_aime.yaml`
- `configs/config_aime24.yaml`
- `configs/config_humaneval.yaml`
- `configs/config_kodcode.yaml`
- `configs/config_math.yaml`
- `configs/config_ot.yaml`
- `configs/config_ot_code.yaml`

All datasets, including the subsetted KodCode dataset, are stored in `datasets/`.

## Executor and Grader Types

The default executor and grader types are specified in the config files. In practice, these settings can be model-specific, so you may need to adjust them for a particular run.

For example, we suggest using the generic extractor for the Qwen 14B model on the AIME datasets due to model response style.

## Notes

The generation scripts and configs are the main entry points for running experiments. A typical workflow is:

1. Install dependencies.
2. Start the code executor.
3. Start the vLLM server.
4. Run the appropriate script from `scripts/generation/`.
5. Adjust the config file in `configs/` if the target model requires a different executor, grader, temperature, or extractor.

## SCaTR Classifier Pipeline

After generating and evaluating responses, the SCaTR pipeline trains lightweight classifiers on intermediate LLM embeddings for best-of-N response selection. The pipeline consists of three sequential steps.

### 1. Parse Responses

```bash
python parse_responses.py \
    --model <model> \
    --dataset <dataset> \
```

Reads raw generation output files, extracts correctness labels (`is_correct`), and structures them into per-turn, per-seed `.pkl` files consumed by downstream steps. Grader and extractor behavior follows the same config-level settings as the generation step — adjust these if the model response style requires it (e.g., use the generic extractor for Qwen 14B on AIME).

### 2. Extract Intermediate Embeddings

```bash
python extracting_embeddings.py \
    --model <model> \
    --dataset <dataset> \
    --layers <layer indices> \
    --embedding-types <desired embedding type>
    --gpu-ids <GPUs to use>
```

Loads the frozen LLM and extracts hidden states from the specified intermediate layers for each parsed response. Outputs are stored as `.pkl` files, one per model/dataset/turn/seed combination. This step is parallelized across GPUs — set available GPU IDs accordingly.

### 3. Train SCaTR Classifiers

```bash
python scatr.py \
    --train <train dataset> \
    --test <test dataset> \
    --model <model> \
    --layer <layer index> \
    --type <embedding type> \
    --model-type <nn|transformer> \
    --n-rollouts <N> \
    --gpu-ids <GPUs to use>
```

Trains MLP or Transformer classifiers on the extracted embeddings and evaluates best-of-N selection performance. Key arguments:

- `--type`: embedding aggregation strategy — `final` (last token), `mean` (mean pool), `last_10`, `attn_weighted`, `special`, or `all`
- `--layer`: comma-separated layer indices to use (e.g. `24` or `12,24,36`)
- `--model-type`: `nn` for MLP classifiers, `transformer` for attention-based classifiers
- `--n-rollouts`: evaluate best-of-N at a fixed rollout count; omit to use all available rollouts
- `--rollouts`: evaluate at n_rollouts ∈ {2, 4, 8, 12, all} in a single run (mutually exclusive with `--n-rollouts`)
- `--grid`: run a random hyperparameter search instead of the default Optuna TPE search

Results are written to `results/`

### Typical SCaTR Workflow

After completing steps 1–5 in the generation workflow above:

6. Run `parse_responses.py` to extract correctness labels from generation outputs.
7. Run `extracting_embeddings.py` to extract intermediate layer representations.
8. Run `scatr.py` to train classifiers and evaluate best-of-N performance.

## Baselines

The LoRA and PRM baselines are in `baselines/`.
