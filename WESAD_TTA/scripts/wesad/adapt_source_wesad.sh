#!/bin/bash

target=(S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S13 S14 S15 S16 S17)

for target_subject in "${target[@]}"
do
  python adapt.py \
    --target_domain "$target_subject" \
    --dataset_cfg ./cfg/dataset/wesad.yaml \
    --algorithm_cfg ./cfg/algorithm/source.yaml
done
