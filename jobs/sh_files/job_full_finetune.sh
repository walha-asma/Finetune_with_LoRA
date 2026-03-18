#!/bin/bash
#SBATCH --job-name=full_finetune
#SBATCH --output=jobs/logs/full_finetune_%j.out
#SBATCH --error=jobs/logs/full_finetune_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --exclude=ensta-l40s01.r2.enst.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_DIR="$HOME/text-in-image-generation/Finetune_with_LoRA"
cd "$PROJECT_DIR"

source "$HOME/text-in-image-generation/venv/bin/activate"

mkdir -p results/metrics results/generated_images

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date   : $(date)"

python -u train_full_finetune.py

echo "Done: $(date)"
