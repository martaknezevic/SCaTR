#!/bin/bash

# Run code evaluator 3 times with different output directories
for i in {1..3}
do
    echo "========================================="
    echo "Running evaluation turn $i/3"
    echo "========================================="
    
    python math_evaluator.py \
        --config configs/config_math.yaml \
        --output_dir "/tmp/scatr/data/qwen1_7b/math500/turn${i}" \
        --temperature 0.6 \
        --base_url "http://localhost:8000/v1"
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on math500 turn $i"
        exit 1
    fi

    python math_evaluator.py \
        --config configs/config_aime.yaml \
        --output_dir "/tmp/scatr/data/qwen1_7b/aime/turn${i}" \
        --temperature 0.6 \
        --base_url "http://localhost:8000/v1"   
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on aime turn $i"
        exit 1
    fi
done

