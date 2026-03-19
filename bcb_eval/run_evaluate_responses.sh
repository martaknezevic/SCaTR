#python evaluate_responses.py --responses /efs/cactts/data/qwen1_7b/bigcodebench_hard/turn1/all_response_metrics_regenerated.jsonl --dataset /home/ubuntu/cactts/datasets/bigcodebench_hard.jsonl --output /efs/cactts/data/qwen1_7b/bigcodebench_hard/turn1/all_response_metrics_regraded.jsonl


#python evaluate_responses.py --responses /efs/cactts/data/gptoss/bigcodebench_hard/turn1/all_response_metrics.jsonl --dataset /home/ubuntu/cactts/datasets/bigcodebench_hard.jsonl --output /efs/cactts/data/gptoss/bigcodebench_hard/turn1/all_response_metrics_regraded_v2.jsonl
#python evaluate_responses.py --responses /efs/cactts/data/gptoss/bigcodebench_hard/turn2/all_response_metrics.jsonl --dataset /home/ubuntu/cactts/datasets/bigcodebench_hard.jsonl --output /efs/cactts/data/gptoss/bigcodebench_hard/turn2/all_response_metrics_regraded_v2.jsonl
#python evaluate_responses.py --responses /efs/cactts/data/gptoss/bigcodebench_hard/turn3/all_response_metrics.jsonl --dataset /home/ubuntu/cactts/datasets/bigcodebench_hard.jsonl --output /efs/cactts/data/gptoss/bigcodebench_hard/turn3/all_response_metrics_regraded_v2.jsonl

cp /efs/cactts/data/gptoss/bigcodebench_hard/turn3/all_response_metrics_regraded_v2.jsonl /efs/cactts/data/gptoss/bigcodebench_hard/turn3/all_response_metrics.jsonl
cp /efs/cactts/data/gptoss/bigcodebench_hard/turn2/all_response_metrics_regraded_v2.jsonl /efs/cactts/data/gptoss/bigcodebench_hard/turn2/all_response_metrics.jsonl
cp /efs/cactts/data/gptoss/bigcodebench_hard/turn1/all_response_metrics_regraded_v2.jsonl /efs/cactts/data/gptoss/bigcodebench_hard/turn1/all_response_metrics.jsonl