#!/usr/bin/env bash
#SBATCH --job-name=nanobrain
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --error=slurms/slurm-%j.err

set -euo pipefail

OUT_DIR="output"
NAME="vits_1hr"
RUN_DIR="${OUT_DIR}/${NAME}"

# run training with a fixed wall time budget
# timeout exits 124 when it stops training, which is expected
status=0
timeout --kill-after=2m "1h" \
    uv run --no-sync python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    exit $status
fi

# run evals if successful, ~10 min
uv run --no-sync python -m nanobrain.eval.run \
    "${RUN_DIR}/checkpoint-last.pth" \
    "${RUN_DIR}/eval"
