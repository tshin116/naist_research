#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

"$PYTHON" compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source norm tent oftta ema_tent mi_ema_tent_mi_gate_only mi_dynamic_ema_tent mi_dynamic_ema_tent_fixed_ema_relaxed \
  --run_name WESAD3class_MI_Dynamic_EMATent \
  "$@"
