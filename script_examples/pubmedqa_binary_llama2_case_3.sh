#!/bin/bash
#SBATCH --chdir=/home/add_your_path/lm-evaluation-harness/
#SBATCH -p short
#SBATCH --gres=gpu:a100.80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=7b_llama2_case3_1
#SBATCH --output=/home/add_your_path/lm-evaluation-harness/output/llama2/7b/pubmedqa_long_binary_llama2-7b_case3_1.txt
#SBATCH --error=/home/add_your_path/lm-evaluation-harness/error/llama2/7b/pubmedqa_long_binary_llama2-7b_case3_1.err
#SBATCH -N 1

pwd

python /home/add_your_path/lm-evaluation-harness/lm_eval/__main__.py --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf --tasks pubmedqa_long_binary_case3 --device cuda:0 --batch_size auto --log_samples --output_path /home/add_your_path/lm-evaluation-harness/output/llama2