#!/usr/bin/env bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --partition=n
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_NAME="baseline"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

name="vits_run1"
fullname="${EXP_NAME}/${name}"

uv run --no-sync python -m nanobrain.train \
    --overrides \
    name="${fullname}" \
    output_dir="${OUT_DIR}" \
    wandb=false
