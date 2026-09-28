#!/usr/bin/env bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --partition=n
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --array=0-31

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_NAME="baseline"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

task_id="${SLURM_ARRAY_TASK_ID}"
name="vits_sweep1_seed${task_id}"
fullname="${EXP_NAME}/${name}"
notes="contention sweep, 16 parallel runs"

uv run --no-sync python -m nanobrain.train \
    --overrides \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}" \
    seed="${task_id}" \
    wandb=false
