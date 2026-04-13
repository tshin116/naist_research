#!/bin/bash

# 固定 epoch 学習済み checkpoint を使って、Source、Tent、OFTTA の比較表とグラフを作成する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  --resume ./ckpt_fixed \
  "$@"
