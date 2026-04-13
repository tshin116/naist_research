#!/bin/bash

# 内側 LOSO を使わず、固定 epoch で WESAD の外側 LOSO 学習を実行する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python train_fixed_loso.py \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  --algorithm_cfg ./cfg/algorithm/source_fixed.yaml
