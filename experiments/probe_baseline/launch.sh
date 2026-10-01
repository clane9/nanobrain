#!/usr/bin/env bash
#SBATCH --job-name=probe_baseline
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --partition=n
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --array=0-5

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

MODELS=(
    raw_voxel
    vits_random_init
    vits_run2_epoch10
    vits_run2_epoch20
    vits_run2_epoch30
    vits_run2_epoch40
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

uv run --no-sync python -m nanobrain.eval.run \
    "experiments/probe_baseline/checkpoints/${MODEL}.pth" \
    "experiments/probe_baseline/output/${MODEL}"
