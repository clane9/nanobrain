#!/usr/bin/env bash
#SBATCH --job-name=baseline_e2e
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=1:15:00
#SBATCH --partition=n
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --account=sophont

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd $ROOT

set -a
source .env
set +a

EXP_NAME="baseline_e2e"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

TRAIN_TIMEOUT="10m"
name="vits_${TRAIN_TIMEOUT}"
fullname="${EXP_NAME}/${name}"
RUN_DIR="${OUT_DIR}/${fullname}"

# timeout exits 124 when it stops training, which is expected
status=0
timeout --kill-after=2m "${TRAIN_TIMEOUT}" \
    uv run --no-sync python -m nanobrain.train \
    --overrides \
    name="${fullname}" \
    output_dir="${OUT_DIR}" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    exit $status
fi

uv run --no-sync python -m nanobrain.eval.run \
    "${RUN_DIR}/checkpoint-last.pth" \
    "${RUN_DIR}/eval"
