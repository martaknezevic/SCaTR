#!/bin/bash

# Run code evaluator 3 times with different output directories
for i in {1..2}
do
    echo "========================================="
    echo "Running evaluation turn $i/2"
    echo "========================================="
    
    python code_generator.py \
        --config config_bigcodebench_hard.yaml \
        --output_dir "/efs/cactts/data/qwen1_7b/bigcodebench_hard/turn${i}" \
        --temperature 0.6 \
        --base_url "http://localhost:8000/v1"
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on humaneval turn $i"
        exit 1
    fi
    
    echo ""
    echo "Turn $i completed successfully"
    echo ""
done

echo "========================================="
echo "All 2 evaluation turns completed!"
echo "========================================="



