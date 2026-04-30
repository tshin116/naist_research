#!/bin/bash

# EMA-Tent入りの通常順序CSVとshuffle CSVを使い、batch bias と TTA性能差の相関を可視化する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env python analyze_batch_bias.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --comparison_csv ./logs/wesad/compare_source_tent_oftta/260420_151858_EMATent手法導入後評価/source_tent_oftta_comparison.csv \
  --shuffle_comparison_csv ./logs/wesad/compare_source_tent_oftta/260420_155012_EMATent手法導入後評価-shuffle/source_tent_oftta_comparison.csv \
  --methods Tent OFTTA EMATent \
  "$@"
