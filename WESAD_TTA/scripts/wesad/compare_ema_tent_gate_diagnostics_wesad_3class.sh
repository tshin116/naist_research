#!/bin/bash

# EMA-Tent gate が保守的すぎるかを検証するための診断実験。
# 出力先には各 batch の raw/effective gate、実効BN混合重み、真のラベル構成が保存される。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA

PYTHON="${PYTHON:-/work/shinsaku-t/miniconda3/envs/wesad_env/bin/python}"

$PYTHON compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source tent ema_tent ema_tent_probe025 ema_tent_probe050 ema_tent_probe100 ema_tent_mingate025 ema_tent_mingate050 \
  --run_name WESAD3class_EMATent_gate_diagnostics_sequential \
  "$@"

$PYTHON compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class_target_shuffle.yaml \
  --resume ./ckpt_3class \
  --methods source tent ema_tent ema_tent_probe025 ema_tent_probe050 ema_tent_probe100 ema_tent_mingate025 ema_tent_mingate050 \
  --run_name WESAD3class_EMATent_gate_diagnostics_shuffle \
  "$@"
