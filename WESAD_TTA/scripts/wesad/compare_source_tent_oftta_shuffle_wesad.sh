#!/bin/bash

# target loader を shuffle して、Source、Tent、OFTTA の比較表とグラフを作成する。
# checkpoint はデフォルトの ./ckpt を使う。固定 epoch 版を使う場合は fixed_shuffle 版を使う。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_target_shuffle.yaml \
  "$@"
