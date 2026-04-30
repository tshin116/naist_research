#!/bin/bash

# EMA-Tent の safe/default/adaptive 設定を通常順序と shuffle 条件で比較する。
# 各比較では提案手法を図の一番右に置くため、methods の最後に EMA-Tent variant を指定する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA

variants=(ema_tent_safe ema_tent ema_tent_adaptive)
dataset_cfgs=(./cfg/dataset/wesad_3class.yaml ./cfg/dataset/wesad_3class_target_shuffle.yaml)
condition_names=(sequential shuffle)

for idx in "${!dataset_cfgs[@]}"
do
  dataset_cfg="${dataset_cfgs[$idx]}"
  condition_name="${condition_names[$idx]}"
  for variant in "${variants[@]}"
  do
    echo "=== EMA-Tent variant: ${variant} (${condition_name}) ==="
    conda run -n wesad_env python compare_source_tent_oftta.py \
      --dataset_cfg "$dataset_cfg" \
      --resume ./ckpt_3class \
      --methods source tent oftta "$variant" \
      "$@"
  done
done
