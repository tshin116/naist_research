#!/bin/bash

# WESAD の 1D-CNN source model を LOSO で学習する入口。
# 詳細な設定は cfg/dataset/wesad.yaml と cfg/algorithm/source.yaml に分離している。
python train.py \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  --algorithm_cfg ./cfg/algorithm/source.yaml
