#python generate_logprobs.py  --model qwen1_7b --dataset humaneval --turn 1 --batch_size 16 --max_concurrent 16 --correct 
#python generate_logprobs.py  --model qwen1_7b --dataset humaneval --turn 2 --batch_size 16 --max_concurrent 16 --correct
#python generate_logprobs.py  --model qwen1_7b --dataset humaneval --turn 3 --batch_size 16 --max_concurrent 16 --correct
#python generate_logprobs.py  --model qwen1_7b --dataset humaneval --turn 1 --batch_size 16 --max_concurrent 16
#python generate_logprobs.py  --model qwen1_7b --dataset humaneval --turn 2 --batch_size 16 --max_concurrent 16
python generate_logprobs.py  --model qwen1_7b --dataset humaneval --turn 3 --batch_size 16 --max_concurrent 16

python generate_logprobs.py  --model qwen1_7b --dataset kodcode --turn 1 --batch_size 16 --max_concurrent 16 --correct
python generate_logprobs.py  --model qwen1_7b --dataset kodcode --turn 2 --batch_size 16 --max_concurrent 16 --correct
python generate_logprobs.py  --model qwen1_7b --dataset kodcode --turn 3 --batch_size 16 --max_concurrent 16 --correct
python generate_logprobs.py  --model qwen1_7b --dataset kodcode --turn 1 --batch_size 16 --max_concurrent 16
python generate_logprobs.py  --model qwen1_7b --dataset kodcode --turn 2 --batch_size 16 --max_concurrent 16
python generate_logprobs.py  --model qwen1_7b --dataset kodcode --turn 3 --batch_size 16 --max_concurrent 16
