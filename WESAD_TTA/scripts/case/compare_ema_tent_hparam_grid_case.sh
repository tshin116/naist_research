#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"

METHODS=(
  source
  norm
  tent
  oftta
  ema_tent_safe
  ema_tent_grid_s40_e50_b10_m085
  ema_tent_grid_s30_e50_b20_m085
  ema_tent_grid_s20_e50_b30_m075
  ema_tent
  ema_tent_adaptive
  ema_tent_grid_s20_e50_b30_m090
  ema_tent_grid_s20_e50_b30_m095
  ema_tent_grid_s10_e50_b40_m085
)

"${PYTHON}" compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_valence.yaml \
  --resume ./ckpt_case_valence \
  --methods "${METHODS[@]}" \
  --run_name CASE_valence_EMATentHparamGrid

"${PYTHON}" compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_arousal.yaml \
  --resume ./ckpt_case_arousal \
  --methods "${METHODS[@]}" \
  --run_name CASE_arousal_EMATentHparamGrid
