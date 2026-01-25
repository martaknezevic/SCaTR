#!/bin/bash



# Run code evaluator 3 times with different output directories
for i in {1..3}
do
    echo "========================================="
    echo "Running evaluation turn $i/3"
    echo "========================================="
    
    python code_evaluator.py \
        --config config_humaneval.yaml \
        --output_dir "/efs/cactts/data/ollmo7b/humaneval/turn${i}" \
        --temperature 1.0 \
        --base_url "http://localhost:9000/v1"
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on humaneval turn $i"
        exit 1
    fi

    
done

echo "========================================="
echo "All 3 evaluation turns completed!"
echo "========================================="
