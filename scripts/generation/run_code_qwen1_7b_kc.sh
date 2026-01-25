#!/bin/bash

# Run code evaluator 3 times with different output directories
for i in {6..7}
do
    echo "========================================="
    echo "Running evaluation turn $i/2"
    echo "========================================="
    
    #python code_evaluator.py \
    #    --config config_humaneval.yaml \
    #    --output_dir "/efs/cactts/data/qwen1_7b/humaneval/turn${i}" \
    #    --temperature 0.6 \
    #    --base_url "http://localhost:8000/v1"
    #
    #if [ $? -ne 0 ]; then
    #    echo "Error: Evaluation failed on humaneval turn $i"
    #    exit 1
    #fi

    python code_generator.py \
        --config config_kodcode.yaml \
        --output_dir "/efs/cactts/data/qwen1_7b/kodcode/turn${i}" \
        --temperature 0.6 \
        --base_url "http://localhost:8000/v1"   
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on kodcode turn $i"
        exit 1
    fi
    
    echo ""
    echo "Turn $i completed successfully"
    echo ""
done

echo "========================================="
echo "All 2 evaluation turns completed!"
echo "========================================="



