#!/bin/bash

# 3分類 checkpoint を使い、target loader を shuffle して Source / Tent / OFTTA / EMA-Tent を比較する。
# 提案手法 EMA-Tent が図の一番右に出るよう、methods の最後に置く。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class_target_shuffle.yaml \
  --resume ./ckpt_3class \
  --methods source tent oftta ema_tent \
  "$@"
