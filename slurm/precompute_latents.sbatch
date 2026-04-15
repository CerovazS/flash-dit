#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_LENS/lcerovaz/flash-dit
#SBATCH --job-name=flash-dit-precompute
#SBATCH --output=./slurm/precompute-%j.out
#SBATCH --error=./slurm/precompute-%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=iscrc_lens

set -euo pipefail

module load cuda/12.2

export HF_TOKEN="$(grep HF_TOKEN ~/.bashrc | grep -oP '(?<=")hf_[^"]+(?=")')"
export http_proxy='http://login01:1909'
export https_proxy='http://login01:1909'
export HUGGINGFACE_HUB_CACHE=/leonardo_scratch/fast/IscrC_LENS/lcerovaz/flash-dit-cache/hf_cache

mkdir -p slurm

source .venv/bin/activate

echo "[$(date)] Starting latent precomputation"
uv run python scripts/precompute_latents.py
echo "[$(date)] Done"
