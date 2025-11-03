#!/bin/bash

MODEL_NAME="Qwen8b"
OUTPUT_DIR="offline_evaluation_results"
CHOICES_BASE="big_results/qwen8b/MathArena_aime_2025"

# topk logprobs sweep
for turn in {1..10}; do
    # Iterate over top-k logprobs
    for k in 10 8 6 4 2; do
        echo "Running evaluation for turn ${turn}, topk_logprobs=${k}..."
        python evaluate.py \
            --choices_dir "${CHOICES_BASE}/turn${turn}/choices" \
            --model_name "${MODEL_NAME}" \
            --turn "${turn}" \
            --topk_logprobs "${k}" \
            --output_dir "${OUTPUT_DIR}"
        echo "Done: turn ${turn}, topk=${k}"
        echo "----------------------------------------"
    done
done

# topk logprobs sweep
for turn in {1..10}; do
    # Iterate over top-k logprobs
    for group_size in 512 2048; do
        echo "Running evaluation for turn ${turn}, group_size=${group_size}..."
        python evaluate.py \
            --choices_dir "${CHOICES_BASE}/turn${turn}/choices" \
            --model_name "${MODEL_NAME}" \
            --turn "${turn}" \
            --topk_logprobs 10 \
            --group_size "${group_size}" \
            --output_dir "${OUTPUT_DIR}"
        echo "Done: turn ${turn}, group_size=${group_size}"
        echo "----------------------------------------"
    done
done

# topk logprobs sweep
for turn in {1..10}; do
    # Iterate over top-k logprobs
    for tail_n in 256 512 1024 4096; do
        echo "Running evaluation for turn ${turn}, tail_n=${tail_n}..."
        python evaluate.py \
            --choices_dir "${CHOICES_BASE}/turn${turn}/choices" \
            --model_name "${MODEL_NAME}" \
            --turn "${turn}" \
            --topk_logprobs 10 \
            --tail_n "${tail_n}" \
            --output_dir "${OUTPUT_DIR}"
        echo "Done: turn ${turn}, tail_n=${tail_n}"
        echo "----------------------------------------"
    done
done

