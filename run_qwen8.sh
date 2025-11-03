
#!/bin/bash

DATASET_SOURCE="MathArena/aime_2025"
OUTPUT_DIR="./big_results/qwen8b"
N_GEN=64
MAX_CONCURRENT=4
BATCH_SIZE=4
PORT=8000
TAIL_N=2048

# Loop over turns 1–10
# python cactts_aime25.py --dataset_source MathArena/aime_2025 --n_gen 8 --max_concurrent 4 --batch_size 4 --turn 1 --port 8000 --output_dir ./big_results/qwen8b --tail_n 2048
for turn in {1..10}; do
    echo "Running CACTTS for turn ${turn}..."
    python cactts_aime25.py \
        --dataset_source "${DATASET_SOURCE}" \
        --n_gen "${N_GEN}" \
        --max_concurrent "${MAX_CONCURRENT}" \
        --batch_size "${BATCH_SIZE}" \
        --turn "${turn}" \
        --port "${PORT}" \
        --output_dir "${OUTPUT_DIR}" \
        --tail_n "${TAIL_N}" \
        
    echo "Completed turn ${turn}"
    echo "----------------------------------------"
done

