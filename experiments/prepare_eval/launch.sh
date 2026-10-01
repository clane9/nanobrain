#!/usr/bin/env bash
#SBATCH --job-name=prepare_eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --partition=n
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

export FOMO_RAW_ROOT="/data/smri-datasets/fomo_eval"

for name in fomo_task1 fomo_task2 fomo_task3 fomo_task4 fomo_task5; do
    uv run --no-sync python -m nanobrain.eval.prepare "${name}" --num-workers 8
done
