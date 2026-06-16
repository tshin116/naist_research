#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"

METHODS=(
  source
  norm
  tent
  oftta
  ema_tent_grid_s20_e50_b30_m075
  ema_tent_case_s20_e50_b30_m060
  ema_tent_case_s20_e50_b30_m065
  ema_tent_case_s20_e50_b30_m070
  ema_tent_case_s10_e50_b40_m070
  ema_tent_case_s10_e50_b40_m075
  ema_tent_case_s10_e50_b40_m080
  ema_tent_case_s00_e50_b50_m070
  ema_tent_case_s00_e50_b50_m075
  ema_tent_case_s00_e50_b50_m080
  ema_tent_case_s10_e40_b50_m075
)

"${PYTHON}" compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_valence.yaml \
  --resume ./ckpt_case_valence \
  --methods "${METHODS[@]}" \
  --run_name CASE_valence_EMATentFocusedGrid

"${PYTHON}" compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_arousal.yaml \
  --resume ./ckpt_case_arousal \
  --methods "${METHODS[@]}" \
  --run_name CASE_arousal_EMATentFocusedGrid
