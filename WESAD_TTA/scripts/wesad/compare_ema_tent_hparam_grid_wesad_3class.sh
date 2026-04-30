#!/bin/bash

# EMA-Tent のハイパーパラメータ寄与を切り分けるための focused grid。
# 生成した cfg/algorithm/ema_tent_grid_*.yaml を使い、各設定を図の一番右に置いて比較する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA

make_cfg() {
  local name="$1"
  local source_weight="$2"
  local ema_weight="$3"
  local batch_weight="$4"
  local momentum="$5"
  local path="cfg/algorithm/${name}.yaml"

  cat > "$path" <<EOF
adaption: ema_tent
tent_lr: 0.0001
tent_steps: 1
episodic: false

ema_tent_source_weight: ${source_weight}
ema_tent_ema_weight: ${ema_weight}
ema_tent_batch_weight: ${batch_weight}
ema_tent_momentum: ${momentum}

ema_tent_use_gate: true
ema_tent_gate_threshold: 0.2
ema_tent_gate_power: 1.0
ema_tent_min_confidence: 0.0
ema_tent_loss_gate: true
ema_tent_probe_gate: 0.0
EOF
}

# 1. weight sweep: momentum は adaptive と同じ 0.85 に固定し、
#    source と current batch のトレードオフを見る。
make_cfg ema_tent_grid_s10_e50_b40_m085 0.1 0.5 0.4 0.85
make_cfg ema_tent_grid_s20_e50_b30_m085 0.2 0.5 0.3 0.85
make_cfg ema_tent_grid_s30_e50_b20_m085 0.3 0.5 0.2 0.85
make_cfg ema_tent_grid_s40_e50_b10_m085 0.4 0.5 0.1 0.85

# 2. momentum sweep: 現時点で最良の adaptive weight に固定し、
#    target EMA の追従速度を見る。
make_cfg ema_tent_grid_s20_e50_b30_m075 0.2 0.5 0.3 0.75
make_cfg ema_tent_grid_s20_e50_b30_m080 0.2 0.5 0.3 0.80
make_cfg ema_tent_grid_s20_e50_b30_m085 0.2 0.5 0.3 0.85
make_cfg ema_tent_grid_s20_e50_b30_m090 0.2 0.5 0.3 0.90
make_cfg ema_tent_grid_s20_e50_b30_m095 0.2 0.5 0.3 0.95

variants=(
  ema_tent_grid_s10_e50_b40_m085
  ema_tent_grid_s20_e50_b30_m085
  ema_tent_grid_s30_e50_b20_m085
  ema_tent_grid_s40_e50_b10_m085
  ema_tent_grid_s20_e50_b30_m075
  ema_tent_grid_s20_e50_b30_m080
  ema_tent_grid_s20_e50_b30_m090
  ema_tent_grid_s20_e50_b30_m095
)

for variant in "${variants[@]}"
do
  echo "=== EMA-Tent hparam grid: ${variant} ==="
  conda run -n wesad_env python compare_source_tent_oftta.py \
    --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
    --resume ./ckpt_3class \
    --methods source tent oftta "$variant" \
    "$@"
done
