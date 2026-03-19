#rm /efs/cactts/data/gptoss/humaneval/turn1/all_response_metrics_regraded_v2.jsonl
#rm /efs/cactts/data/gptoss/humaneval/turn2/all_response_metrics_regraded_v2.jsonl
#rm /efs/cactts/data/gptoss/humaneval/turn3/all_response_metrics_regraded_v2.jsonl
#
#rm /efs/cactts/data/gptoss/kodcode/turn1/all_response_metrics_regraded_v2.jsonl
#rm /efs/cactts/data/gptoss/kodcode/turn2/all_response_metrics_regraded_v2.jsonl
#rm /efs/cactts/data/gptoss/kodcode/turn3/all_response_metrics_regraded_v2.jsonl
#
#python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics /efs/cactts/data/gptoss/humaneval/turn1/all_response_metrics.jsonl --loader humaneval --max-concurrent 64 --timeout 10 --output /efs/cactts/data/gptoss/humaneval/turn1/all_response_metrics_regraded_v2.jsonl
#python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics /efs/cactts/data/gptoss/humaneval/turn2/all_response_metrics.jsonl --loader humaneval --max-concurrent 64 --timeout 10 --output /efs/cactts/data/gptoss/humaneval/turn2/all_response_metrics_regraded_v2.jsonl
#python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics /efs/cactts/data/gptoss/humaneval/turn3/all_response_metrics.jsonl --loader humaneval --max-concurrent 64 --timeout 10 --output /efs/cactts/data/gptoss/humaneval/turn3/all_response_metrics_regraded_v2.jsonl
#
#python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics /efs/cactts/data/gptoss/kodcode/turn1/all_response_metrics.jsonl --loader kodcode --max-concurrent 128 --timeout 10 --output /efs/cactts/data/gptoss/kodcode/turn1/all_response_metrics_regraded_v2.jsonl
#python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics /efs/cactts/data/gptoss/kodcode/turn2/all_response_metrics.jsonl --loader kodcode --max-concurrent 128 --timeout 10 --output /efs/cactts/data/gptoss/kodcode/turn2/all_response_metrics_regraded_v2.jsonl
#python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics /efs/cactts/data/gptoss/kodcode/turn3/all_response_metrics.jsonl --loader kodcode --max-concurrent 128 --timeout 10 --output /efs/cactts/data/gptoss/kodcode/turn3/all_response_metrics_regraded_v2.jsonl


