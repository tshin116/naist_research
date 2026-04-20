#!/bin/bash

# WESAD 3分類の target batch 内クラス分布と TTA 性能差の関係を可視化する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env python analyze_batch_bias.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --comparison_csv ./logs/wesad/compare_source_tent_oftta/260419_212826_Tent2回順伝搬廃止正規評価/source_tent_oftta_comparison.csv \
  --shuffle_comparison_csv ./logs/wesad/compare_source_tent_oftta/260420_071212_Tent2回順伝搬廃止正規評価shuffle/source_tent_oftta_comparison.csv \
  "$@"
