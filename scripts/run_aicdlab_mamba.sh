#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  echo "Could not find Miniforge at $HOME/miniforge3." >&2
  exit 1
fi

conda activate "${ENV_NAME:-aicdlab-mamba}"
cd "${PROJECT_DIR:-$HOME/AICDLab1}"

python scripts/prepare_splits.py --data-dir Data --classes cats dogs --folds 5

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-30}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-animal_binary_mamba}"
WORKERS="${WORKERS:-4}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-1}"

python train.py \
  --train-csv Data/folds/fold_0_train.csv \
  --val-csv Data/folds/fold_0_val.csv \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --workers "$WORKERS" \
  --freeze-backbone-epochs "$FREEZE_BACKBONE_EPOCHS" \
  --amp \
  --use-randaugment
