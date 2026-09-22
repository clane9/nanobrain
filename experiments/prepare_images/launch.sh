#!/usr/bin/env bash
#SBATCH --job-name=prepare_images
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=c
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont
#SBATCH --qos=high

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_DIR="experiments/prepare_images"
OUT_DIR="data/FOMO300K_images"

parallel --jobs 64 \
    uv run --no-sync python -m nanobrain.prepare_images {} \
    --out-root "${OUT_DIR}" \
    ::: {0..63}
