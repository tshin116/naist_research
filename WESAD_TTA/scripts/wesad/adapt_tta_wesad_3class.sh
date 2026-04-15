#!/bin/bash

# 3分類 checkpoint を使い、source と各 TTA 手法を全被験者で評価する。
target=(S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S13 S14 S15 S16 S17)
methods=(source norm tent pl shot sar t3a tast tast_bn oftta)

cd /work/shinsaku-t/naist_reserch/WESAD_TTA

for method in "${methods[@]}"
do
  for target_subject in "${target[@]}"
  do
    python adapt.py \
      --target_domain "$target_subject" \
      --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
      --algorithm_cfg "./cfg/algorithm/${method}.yaml"
  done
done
