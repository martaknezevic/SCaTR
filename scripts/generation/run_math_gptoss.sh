#!/bin/bash

# Run code evaluator 3 times with different output directories
for i in {1..3}
do
    echo "========================================="
    echo "Running evaluation turn $i/3"
    echo "========================================="
    
    python math_evaluator.py \
        --config config_math.yaml \
        --output_dir "/efs/cactts/data/gptoss/math500/turn${i}" \
        --temperature 1.0 \
        --base_url "http://localhost:8000/v1"
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on math500 turn $i"
        exit 1
    fi

    python math_evaluator.py \
        --config config_aime.yaml \
        --output_dir "/efs/cactts/data/gptoss/aime/turn${i}" \
        --temperature 1.0 \
        --base_url "http://localhost:8000/v1"   
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed on aime turn $i"
        exit 1
    fi
    
    echo ""
    echo "Turn $i completed successfully"
    echo ""
done

echo "========================================="
echo "All 3 evaluation turns completed!"
echo "========================================="
