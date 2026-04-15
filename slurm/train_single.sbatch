#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_LENS/lcerovaz/flash-dit
#SBATCH --job-name=${RUN_NAME:-flash-dit-train}
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=iscrc_lens

set -euo pipefail

module load cuda/12.2

export HF_TOKEN="$(grep HF_TOKEN ~/.bashrc | grep -oP '(?<=")hf_[^"]+(?=")')"
export http_proxy='http://login01:3128'
export https_proxy='http://login01:3128'

# Required: RUN_NAME, MODEL (default: flash_dit)
RUN_NAME="${RUN_NAME:-flash_dit_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-flash_dit}"
OPTIMIZER="${OPTIMIZER:-muon}"

mkdir -p slurm

source .venv/bin/activate

echo "[$(date)] Starting training — run_name=${RUN_NAME} model=${MODEL}"

srun uv run python scripts/train.py \
    model="${MODEL}" \
    optimizer="${OPTIMIZER}" \
    run_name="${RUN_NAME}"

echo "[$(date)] Training complete"
