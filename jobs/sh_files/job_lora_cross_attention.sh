#!/bin/bash
#SBATCH --job-name=lora_cross_attn
#SBATCH --output=jobs/logs/lora_cross_attn_%j.out
#SBATCH --error=jobs/logs/lora_cross_attn_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --exclude=ensta-l40s01.r2.enst.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_DIR="$HOME/text-in-image-generation/Finetune_with_LoRA"
cd "$PROJECT_DIR"

source "$HOME/text-in-image-generation/venv/bin/activate"

mkdir -p results/metrics results/generated_images

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date   : $(date)"

python -u train_lora_cross_attention.py

echo "Done: $(date)"
