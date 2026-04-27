#!/bin/bash

# Run code evaluator 3 times with different output directories
for i in {1..3}
do
    echo "========================================="
    echo "Running evaluation turn $i/2"
    echo "========================================="
    
    python code_evaluator.py \
        --config configs/config_humaneval.yaml \
        --output_dir "/tmp/scatr/data/qwen1_7b/humaneval/turn${i}" \
        --temperature 0.6 \
        --base_url "http://localhost:8000/v1"
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on humaneval turn $i"
        exit 1
    fi

    python code_evaluator.py \
        --config configs/config_kodcode.yaml \
        --output_dir "/tmp/scatr/data/qwen1_7b/kodcode/turn${i}" \
        --temperature 0.6 \
        --base_url "http://localhost:8000/v1"   
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on kodcode turn $i"
        exit 1
    fi
    
done




