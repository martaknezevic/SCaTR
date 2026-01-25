#python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics /efs/cactts/data/ollmo7b/humaneval/turn1/all_response_metrics.jsonl --loader humaneval --max-concurrent 64 --timeout 10 --output /efs/cactts/data/ollmo7b/humaneval/turn1/all_response_metrics_regraded.jsonl
#python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics /efs/cactts/data/ollmo7b/humaneval/turn2/all_response_metrics.jsonl --loader humaneval --max-concurrent 64 --timeout 10 --output /efs/cactts/data/ollmo7b/humaneval/turn2/all_response_metrics_regraded.jsonl
#python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics /efs/cactts/data/ollmo7b/humaneval/turn3/all_response_metrics.jsonl --loader humaneval --max-concurrent 64 --timeout 10 --output /efs/cactts/data/ollmo7b/humaneval/turn3/all_response_metrics_regraded.jsonl
#python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics /efs/cactts/data/ollmo7b/kodcode/turn1/all_response_metrics.jsonl --loader kodcode --max-concurrent 64 --timeout 10 --output /efs/cactts/data/ollmo7b/kodcode/turn1/all_response_metrics_regraded.jsonl
## python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics /efs/cactts/data/ollmo7b/kodcode/turn2/all_response_metrics.jsonl --loader kodcode --max-concurrent 128 --timeout 10 --output /efs/cactts/data/ollmo7b/kodcode/turn2/all_response_metrics_regraded.jsonl
## python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics /efs/cactts/data/ollmo7b/kodcode/turn3/all_response_metrics.jsonl --loader kodcode --max-concurrent 128 --timeout 10 --output /efs/cactts/data/ollmo7b/kodcode/turn3/all_response_metrics_regraded.jsonl
## #python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics /efs/cactts/data/qwen1_7b/humaneval/turn1/all_response_metrics.jsonl --loader humaneval --max-concurrent 64 --timeout 10 --output /efs/cactts/data/qwen1_7b/humaneval/turn1/all_response_metrics_regraded.jsonl
## #python regrade_responses.py --dataset datasets/humaneval.jsonl --metrics /efs/cactts/data/qwen1_7b/humaneval/turn2/all_response_metrics.jsonl --loader humaneval --max-concurrent 64 --timeout 10 --output /efs/cactts/data/qwen1_7b/humaneval/turn2/all_response_metrics_regraded.jsonl
## python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics /efs/cactts/data/qwen1_7b/kodcode/turn1/all_response_metrics.jsonl --loader kodcode --max-concurrent 128 --timeout 10 --output /efs/cactts/data/qwen1_7b/kodcode/turn1/all_response_metrics_regraded.jsonl
## python regrade_responses.py --dataset datasets/kodcode_1000.jsonl --metrics /efs/cactts/data/qwen1_7b/kodcode/turn2/all_response_metrics.jsonl --loader kodcode --max-concurrent 128 --timeout 10 --output /efs/cactts/data/qwen1_7b/kodcode/turn2/all_response_metrics_regraded.jsonl

### rm -rf /efs/cactts/data/ollmo7b/kodcode
### rm -rf /efs/cactts/data/qwen1_7b/kodcode
### rm -rf /efs/cactts/data/qwen1_7b/humaneval
### rm -rf /efs/cactts/data/ollmo7b/humaneval

#ls /efs/cactts/data/gptoss/kodcode/turn1/choices/ | wc -l
#ls /efs/cactts/data/gptoss/kodcode/turn2/choices/ | wc -l
#ls /efs/cactts/data/gptoss/humaneval/turn1/choices/ | wc -l
#ls /efs/cactts/data/gptoss/humaneval/turn2/choices/ | wc -l


ls /efs/cactts/data/ollmo7b/kodcode/turn1/choices/ | wc -l
ls /efs/cactts/data/ollmo7b/kodcode/turn2/choices/ | wc -l
ls /efs/cactts/data/qwen1_7b/kodcode/turn1/choices/ | wc -l
ls /efs/cactts/data/qwen1_7b/kodcode/turn2/choices/ | wc -l

ls /efs/cactts/data/ollmo7b/humaneval/turn1/choices/ | wc -l
ls /efs/cactts/data/ollmo7b/humaneval/turn2/choices/ | wc -l
ls /efs/cactts/data/qwen1_7b/humaneval/turn1/choices/ | wc -l
ls /efs/cactts/data/qwen1_7b/humaneval/turn2/choices/ | wc -l