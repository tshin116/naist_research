#!/bin/bash

# 学習済み LOSO checkpoint に対して、適応なし評価と Tent 評価を続けて実行する。
# checkpoint が未作成の場合は、先に train.sh を実行する必要がある。
bash scripts/wesad/adapt_source_wesad.sh
bash scripts/wesad/adapt_norm_wesad.sh
bash scripts/wesad/adapt_tent_wesad.sh
bash scripts/wesad/adapt_pl_wesad.sh
bash scripts/wesad/adapt_shot_wesad.sh
bash scripts/wesad/adapt_sar_wesad.sh
bash scripts/wesad/adapt_t3a_wesad.sh
bash scripts/wesad/adapt_tast_wesad.sh
bash scripts/wesad/adapt_tast_bn_wesad.sh
bash scripts/wesad/adapt_oftta_wesad.sh
