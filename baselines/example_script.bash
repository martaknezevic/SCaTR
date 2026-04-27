CUDA_VISIBLE_DEVICES=1 python train_sft_v2.py --train humaneval --test kodcode --model qwen1_7b --bs 1 -- bs_eval 32 --grad_accum 16 --sequential
CUDA_VISIBLE_DEVICES=1 python evaluate_prm.py --dataset aime --model qwen1_7b --turn 1 --bs 1 --max_length 2048 --inference_estimate
