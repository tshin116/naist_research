#!/bin/bash

# 3分類 checkpoint を使い、Memory-balanced OFTTA を各被験者で実行する。
target=(S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S13 S14 S15 S16 S17)

cd /work/shinsaku-t/naist_reserch/WESAD_TTA

for target_subject in "${target[@]}"
do
  conda run -n wesad_env python adapt.py \
    --target_domain "$target_subject" \
    --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
    --algorithm_cfg ./cfg/algorithm/mem_oftta.yaml \
    --resume ./ckpt_3class \
    "$@"
done
