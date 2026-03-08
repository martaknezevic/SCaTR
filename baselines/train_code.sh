python train_sft.py --train humaneval --test kodcode --model qwen1_7b --bs 1 --grad-accum 16

python train_sft.py --train humaneval --test kodcode --model gptoss --bs 4 --grad-accum 4

python train_sft.py --train humaneval --test kodcode --model gptoss --bs 4 --grad-accum 4
