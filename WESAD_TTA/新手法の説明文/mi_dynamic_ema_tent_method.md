# MI-gated Dynamic EMA-Tent

## 目的

MI-gated Dynamic EMA-Tent は、既存の EMA-Tent を壊さずに追加した改良版の
test-time adaptation 手法である。対象は WESAD のようなウェアラブル生理信号の
未知被験者感情推定であり、通常順序の target stream に含まれるブロック構造、
特に単一クラスに偏った batch が連続する状況で、BN 統計と Tent 更新が不安定に
なる問題を抑えることを目的とする。

## 既存手法との位置付け

### Tent

Tent は test batch の BatchNorm 統計を使いながら、予測エントロピーが小さくなる
ように BatchNorm affine パラメータ、つまり gamma と beta のみを更新する。
Conv 層や Linear 層は更新しない。

WESAD の通常順序では、batch が単一クラスに偏りやすいため、現在 batch 統計を
強く使うこと自体が不安定になる可能性がある。

### OFTTA

OFTTA は source 側の情報を anchor として使い、test batch への過度な適応を抑える
設計である。HAR のように batch 構成が比較的安定している場合には有効だったが、
WESAD 通常順序ではブロック構造により batch 統計が偏りやすく、性能改善が限定的に
なる可能性がある。

### 既存 EMA-Tent

既存 EMA-Tent は BN 統計を以下の 3 成分で混合する。

```text
source statistics
target EMA statistics
current batch statistics
```

さらに、batch 平均予測分布の entropy に基づく gate によって current batch 統計の
寄与を調整する。

ただし、既存 gate は batch 平均予測分布の多様性のみを見るため、次の 2 つを区別しにくい。

```text
1. 本当に複数クラスが混在している batch
2. ドメインシフトで全サンプルに対してモデルが迷っている batch
```

## 提案手法

MI-gated Dynamic EMA-Tent では、既存 EMA-Tent の source / EMA / current batch の
3成分混合を維持しつつ、gate を mutual information 型に変更する。

各サンプルの softmax 出力を `p_i`、batch 平均予測分布を `p_bar` とする。

```text
p_bar = (1 / B) sum_i p_i
```

batch 全体の予測多様性:

```text
H(p_bar) = - sum_c p_bar_c log p_bar_c
```

各サンプルの平均不確実性:

```text
(1 / B) sum_i H(p_i)
```

MI gate:

```text
g_MI = ( H(p_bar) - (1 / B) sum_i H(p_i) ) / log C
```

実装では `g_MI` を `[0, 1]` に clamp する。

この gate は、batch 全体では複数クラスに分散しており、かつ各サンプルの予測が
比較的確信的な場合に大きくなる。逆に、単一クラス batch や、モデルが全体的に
迷っている batch では小さくなる。

## 動的混合重み

現 batch 統計の最大寄与を `w_batch_max` とする。デフォルトは `0.3`。

```text
w_batch^t = w_batch_max * g_MI^t
```

信頼できる target batch をどれだけ見てきたかを表す有効サンプル数を保持する。

```text
N_eff^t = N_eff^(t-1) + B * g_MI^t
rho_t = 1 - exp(-N_eff^t / K)
```

残りの重みを source と EMA に分配する。

```text
w_ema^t = (1 - w_batch^t) * rho_t
w_source^t = (1 - w_batch^t) * (1 - rho_t)
```

したがって、初期は source 統計を強く使い、信頼できる target batch を見ていくほど
EMA 統計を強く使う。current batch 統計は `g_MI` が大きい場合だけ強く使われる。

## Gate-aware EMA

EMA 統計の更新率も `g_MI` で制御する。

```text
alpha_t = (1 - m) * g_MI^t
```

```text
mu_ema^t = (1 - alpha_t) mu_ema^(t-1) + alpha_t mu_batch^t
var_ema^t = (1 - alpha_t) var_ema^(t-1) + alpha_t var_batch^t
```

信頼できない batch では `alpha_t` が小さくなり、EMA 統計への蓄積が抑えられる。

## Entropy update

Tent と同様に、エントロピー最小化で更新するのは BN affine パラメータのみである。

```text
update target: BatchNorm gamma, beta
frozen: Conv, Linear
```

ただし、`g_MI < gate_threshold` の場合は entropy update を行わない。デフォルトは
`gate_threshold: 0.05` である。

## 実装上の流れ

1. probe forward を行う
2. softmax 出力から `g_MI` を計算する
3. `g_MI` から source / EMA / current batch の混合重みを決める
4. 混合 BN 統計で adapt forward を行う
5. adapt forward の logits を評価に使う
6. `g_MI >= gate_threshold` のときだけ entropy update を行う
7. `alpha_t = (1 - m) * g_MI` で EMA 統計を更新する

TTA 適応計算にはラベルを使わない。ラベル分布は評価後の診断ログにのみ保存する。

## 実装ファイル

```text
TTA/adapt_algorithm/mi_dynamic_ema_tent.py
TTA/setup.py
cfg/algorithm/mi_dynamic_ema_tent.yaml
cfg/algorithm/mi_ema_tent_mi_gate_only.yaml
cfg/algorithm/mi_ema_tent_gate_aware.yaml
cfg/algorithm/mi_ema_tent_dynamic.yaml
```

## アブレーション

```text
ema_tent
  既存 EMA-Tent

mi_ema_tent_mi_gate_only
  gate のみ MI 型に変更
  EMA 更新は固定 momentum
  混合重みは固定ルール

mi_ema_tent_gate_aware
  MI gate
  gate-aware EMA
  混合重みは固定ルール

mi_dynamic_ema_tent
  MI gate
  gate-aware EMA
  dynamic mixture weights
```

## 診断ログ

`compare_source_tent_oftta.py` で実行すると、以下が保存される。

```text
ema_tent_batch_diagnostics.csv
ema_tent_gate_summary.md
tta_stream_summary.csv
```

主な列:

```text
g_mi
diversity_norm
sample_entropy_norm
w_source
w_ema
w_batch
alpha
entropy_updated
loss_entropy
pred_class_*_ratio
true_class_*_ratio
true_max_class_ratio
true_imbalance
```

## 実行例

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
bash scripts/wesad/compare_mi_dynamic_ema_tent_wesad_3class.sh
```

synthetic block stream:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
BLOCK_LENGTHS="16 32 64 128 256 512" \
bash scripts/wesad/compare_mi_dynamic_ema_tent_synthetic_blocks_wesad_3class.sh
```

## Smoke test について

前回の報告で使った smoke test とは、論文用実験ではなく、小さい動作確認のことである。
今回は S2 だけで実行し、以下を確認した。

```text
プログラムが落ちない
MI gate が計算される
BN affine 更新が実行またはskipされる
診断CSVが保存される
stream summaryが保存される
```

したがって smoke test の性能値は手法比較の結論には使わない。

## 初期比較結果

以下は WESAD 3class、通常順序、全被験者評価での初期比較である。

実行条件:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source norm tent oftta ema_tent mi_dynamic_ema_tent \
  --run_name WESAD3class_SourceNormTentOFTTAEMATent_MIDynamicEMATent
```

結果保存先:

```text
logs/wesad/compare_source_tent_oftta/260616_133411_WESAD3class_SourceNormTentOFTTAEMATent_MIDynamicEMATent/
```

平均性能:

| Method | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Source | 0.6915 | 0.5285 |
| Norm | 0.4554 | 0.3972 |
| Tent | 0.4553 | 0.3970 |
| OFTTA | 0.5231 | 0.4499 |
| EMA-Tent | 0.7075 | 0.6136 |
| MI-gated Dynamic EMA-Tent | 0.6927 | 0.5395 |

被験者別に最も高い Macro-F1 を出した回数:

| Method | Wins |
| --- | ---: |
| Source | 2 |
| Norm | 0 |
| Tent | 0 |
| OFTTA | 1 |
| EMA-Tent | 7 |
| MI-gated Dynamic EMA-Tent | 5 |

この初期設定では、平均 Macro-F1 は既存 EMA-Tent が最も高い。
MI-gated Dynamic EMA-Tent は Source よりわずかに高く、Tent/OFTTA よりは明確に高いが、
既存 EMA-Tent には届いていない。

一方で、被験者別では MI-gated Dynamic EMA-Tent が最良となる被験者もあり、
S11、S14、S17、S4、S8 などでは既存 EMA-Tent より高い Macro-F1 を示した。
これは、MI gate によって過度な current batch 利用や不確かな entropy update を抑える
方向が一部被験者では有効であることを示唆する。

ただし、S2、S7、S10、S16 などでは既存 EMA-Tent より大きく低下した。
診断ログでは MI-gated Dynamic EMA-Tent の平均 `g_mi` は 0.1013 と低く、
平均 `w_batch` は 0.0304、平均 `alpha` は 0.0203 であった。
つまり、現在の初期設定はかなり保守的であり、target EMA への蓄積と current batch 統計の
利用を抑えすぎている可能性がある。

現時点での解釈:

```text
Tent / Norm:
  WESAD通常順序では大きく低下。current batch BN統計がブロック構造に引きずられている可能性が高い。

OFTTA:
  Source anchorにより過適応は抑えられるが、WESADでは十分な改善に至らない。

既存 EMA-Tent:
  平均性能は最良。source / EMA / current batch の3成分混合がWESADでは有効。

MI-gated Dynamic EMA-Tent:
  Tent/OFTTAよりは良いが、初期設定では既存EMA-Tentより低い。
  MI gateが厳しすぎ、batch統計とEMA蓄積を抑えすぎている可能性がある。
```

次に検証すべき設定:

```text
gate_threshold を 0.05 から 0.0 または 0.02 に下げる
w_batch_max を 0.3 から 0.4 または 0.5 に上げる
rho_k を 512 から 128 または 256 に下げ、EMAへの移行を速くする
mi_probe_batch_weight を 0.0 から 0.2 程度にして、probe時にsource寄りすぎる問題を緩和する
```

## 追加分析: 初期版が弱かった原因

初期版の診断ログから、性能低下の主因は以下だと判断した。

```text
MI gate 自体が悪いのではなく、
MI gate を EMA 更新率と dynamic mixture weights にそのまま使ったため、
source 統計へ寄りすぎた。
```

初期版の平均診断値:

| Metric | Value |
| --- | ---: |
| g_mi mean | 0.1013 |
| w_source mean | 0.9283 |
| w_ema mean | 0.0413 |
| w_batch mean | 0.0304 |
| alpha mean | 0.0203 |
| entropy update rate | 0.4579 |

この状態では、MI gate が小さい batch で current batch 統計を抑えることには成功しているが、
target EMA への蓄積も弱くなりすぎる。その結果、実質的に source 統計へ戻りすぎ、
未知被験者への適応が不足する。

## 改善実験

次の追加設定を作成して比較した。

```text
mi_dynamic_ema_tent_relaxed
  gate_threshold = 0.0
  mi_gate_power = 0.5
  mi_probe_batch_weight = 0.2
  w_batch_max = 0.4
  rho_k = 128

mi_dynamic_ema_tent_relaxed_anchor
  relaxed に加えて mi_min_gate = 0.1

mi_dynamic_ema_tent_fixed_ema_relaxed
  dynamic mixture weights は使う
  ただし EMA 更新率は gate-aware にせず、固定 momentum に戻す
```

`mi_dynamic_ema_tent_fixed_ema_relaxed` の config:

```yaml
adaption: mi_dynamic_ema_tent
tent_lr: 0.0001
tent_steps: 1
episodic: false

ema_tent_source_weight: 0.2
ema_tent_ema_weight: 0.5
ema_tent_batch_weight: 0.3
ema_momentum: 0.80

mi_gate: true
gate_aware_ema: false
dynamic_mixture_weights: true
gate_threshold: 0.0
mi_loss_gate: true
mi_probe_batch_weight: 0.2
mi_gate_power: 0.5
mi_min_gate: 0.1
w_batch_max: 0.4
rho_k: 128
```

## 改善後の比較結果

実行条件:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source ema_tent mi_ema_tent_mi_gate_only mi_dynamic_ema_tent_relaxed_anchor mi_dynamic_ema_tent_fixed_ema_relaxed \
  --run_name WESAD3class_MI_Dynamic_fixedEMA_relaxed_analysis
```

結果保存先:

```text
logs/wesad/compare_source_tent_oftta/260616_134625_WESAD3class_MI_Dynamic_fixedEMA_relaxed_analysis/
```

平均性能:

| Method | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Source | 0.6915 | 0.5285 |
| EMA-Tent | 0.7075 | 0.6136 |
| MI gate only | 0.6883 | 0.5995 |
| MI dynamic relaxed anchor | 0.7006 | 0.5783 |
| MI dynamic fixed-EMA relaxed | 0.7150 | 0.6278 |

被験者別に最良 Macro-F1 を出した回数:

| Method | Wins |
| --- | ---: |
| Source | 4 |
| EMA-Tent | 3 |
| MI gate only | 1 |
| MI dynamic relaxed anchor | 3 |
| MI dynamic fixed-EMA relaxed | 4 |

`mi_dynamic_ema_tent_fixed_ema_relaxed` は、平均 Macro-F1 で既存 EMA-Tent を
`+0.0142` 上回った。

## 改善後の解釈

追加実験から、以下がわかった。

```text
1. MI gate only は既存 EMA-Tent に近い性能を出した。
   したがって MI gate の発想自体は破綻していない。

2. gate-aware EMA は性能を下げた。
   WESAD の通常順序では単一クラス batch が多く、g_MI が小さい batch が多い。
   そのため EMA 更新率まで小さくすると、target EMA が十分に育たない。

3. dynamic mixture weights は、gateを緩和し、EMA更新を固定に戻すと有効になった。
   current batch 統計の使いすぎは避けつつ、target EMA には安定して情報を蓄積できるためである。
```

現時点の推奨設定は以下である。

```text
cfg/algorithm/mi_dynamic_ema_tent_fixed_ema_relaxed.yaml
```

この設定は、MI gate によって current batch 統計の直接利用は制御するが、
EMA 統計への蓄積は固定 momentum で維持する。WESAD では「batch が偏っているから
現在 batch 統計をそのまま信じない」ことは重要だが、「target EMA の蓄積まで止める」
ことは過度に保守的だったと考えられる。

## WESAD 部品別アブレーション

WESAD 3class、通常順序、全被験者評価で、MI gate、gate-aware EMA、dynamic mixture
weights の寄与を分けて確認した。

実行条件:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source ema_tent mi_ema_tent_mi_gate_only_relaxed mi_ema_tent_gate_aware_relaxed mi_dynamic_ema_tent_fixed_ema_relaxed mi_dynamic_ema_tent_relaxed_anchor \
  --run_name WESAD3class_MI_component_ablation_relaxed
```

結果保存先:

```text
logs/wesad/compare_source_tent_oftta/260616_142425_WESAD3class_MI_component_ablation_relaxed/
```

平均性能:

| Method | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Source | 0.6915 | 0.5285 |
| EMA-Tent | 0.7075 | 0.6136 |
| MI gate only relaxed | 0.6807 | 0.5938 |
| MI gate + gate-aware EMA relaxed | 0.6932 | 0.5676 |
| MI dynamic fixed-EMA relaxed | 0.7150 | 0.6278 |
| MI dynamic + gate-aware EMA relaxed | 0.7006 | 0.5783 |

平均 Macro-F1 の差分:

| Comparison | Mean delta | Wins |
| --- | ---: | ---: |
| MI gate only relaxed - EMA-Tent | -0.0198 | 2/15 |
| Gate-aware EMA relaxed - MI gate only relaxed | -0.0262 | 8/15 |
| Dynamic fixed-EMA relaxed - MI gate only relaxed | +0.0340 | 13/15 |
| Dynamic + gate-aware EMA relaxed - Dynamic fixed-EMA relaxed | -0.0495 | 8/15 |
| Dynamic fixed-EMA relaxed - EMA-Tent | +0.0142 | 10/15 |

この結果から、WESAD では以下の順に解釈できる。

```text
MI gate only:
  既存EMA-Tentより平均Macro-F1が低い。
  予測多様性と不確実性を分離する発想は有効だが、gateだけを差し替えても十分ではない。

gate-aware EMA:
  平均ではさらに悪化した。
  WESAD通常順序では単一クラス寄りbatchが多く、g_MIが小さくなりやすい。
  そのためEMA更新まで抑えると、target統計が蓄積されず、適応が弱くなる。

dynamic mixture weights:
  fixed EMA更新と組み合わせると最も有効だった。
  current batch統計の直接利用はgateで抑えつつ、EMA統計には固定momentumでtarget情報を蓄積できる。
  これにより、単一batchへの過度な追従とsource統計への戻りすぎの両方を避けられる。
```

診断ログでは、真の batch 最大クラス比率の平均が `0.9229` であり、
WESAD 通常順序では batch がかなり単一クラスに偏っている。
この条件では、`g_MI` によって current batch 統計を抑えること自体は妥当だが、
EMA 更新率まで `g_MI` に連動させると保守的すぎる。

したがって、現時点で論文に載せる主張としては、

```text
一番有効だった要素は dynamic mixture weights である。
ただし、EMA更新までgate-awareにするのではなく、EMA更新は固定momentumで維持する方がWESADでは有効だった。
```

とまとめるのが妥当である。

## DynaMix EMA-Tent

部品別アブレーションでは、MI gate や gate-aware EMA よりも、混合重みを動的に
決める要素が最も有効だった。そこで、元の EMA-Tent を基準にし、以下だけを追加した
設定を `DynaMix EMA-Tent` と呼ぶ。

```text
元EMA-Tent:
  batch平均予測分布のentropy gate
  固定momentumによるEMA統計更新
  固定的なsource/EMA/current batch混合

DynaMix EMA-Tent:
  batch平均予測分布のentropy gateは維持
  固定momentumによるEMA統計更新も維持
  source/EMA/current batchの混合重みだけを動的化
```

この設定では、MI gate は使わず、EMA 更新率も gate-aware にしない。
つまり、元 EMA-Tent からの差分は dynamic mixture weights のみである。

設定ファイル:

```text
cfg/algorithm/dynamix_ema_tent.yaml
```

実行条件:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source tent oftta ema_tent dynamix_ema_tent \
  --run_name WESAD3class_TentOFTTAEMATent_DynaMixEMATent
```

結果保存先:

```text
logs/wesad/compare_source_tent_oftta/260616_143909_WESAD3class_TentOFTTAEMATent_DynaMixEMATent/
```

平均性能:

| Method | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Tent | 0.4553 | 0.3970 |
| OFTTA | 0.5231 | 0.4499 |
| EMA-Tent | 0.7075 | 0.6136 |
| DynaMix EMA-Tent | 0.7201 | 0.6172 |

被験者別に最良 Macro-F1 を出した回数:

| Method | Wins |
| --- | ---: |
| Tent | 0 |
| OFTTA | 1 |
| EMA-Tent | 4 |
| DynaMix EMA-Tent | 10 |

DynaMix EMA-Tent は EMA-Tent に対して平均 Macro-F1 で `+0.0036` と改善幅は小さいが、
15被験者中10被験者で EMA-Tent を上回った。一方で、S16 のように EMA-Tent より大きく
低下する被験者もあるため、単純な全面改善ではない。

論文上の主張としては、以下が妥当である。

```text
MI gate や gate-aware EMA は平均性能を改善しなかった。
一方で、source/EMA/current batch の混合重みをtarget streamの進行に応じて動的に
変える DynaMix EMA-Tent は、平均Macro-F1と被験者別勝ち数の両方でEMA-Tentを上回った。
```

図:

```text
tent_oftta_ema_tent_dynamix_macro_f1.pdf
tent_oftta_ema_tent_dynamix_subject_macro_f1.pdf
```

## CASE への適用結果

WESAD で有効だった `mi_dynamic_ema_tent_fixed_ema_relaxed` を CASE の arousal /
valence に適用した。

実行条件:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_arousal.yaml \
  --resume ./ckpt_case_arousal \
  --methods source norm tent oftta ema_tent mi_ema_tent_mi_gate_only mi_dynamic_ema_tent_fixed_ema_relaxed \
  --run_name CASE_arousal_MI_Dynamic_fixedEMA_relaxed

PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_valence.yaml \
  --resume ./ckpt_case_valence \
  --methods source norm tent oftta ema_tent mi_ema_tent_mi_gate_only mi_dynamic_ema_tent_fixed_ema_relaxed \
  --run_name CASE_valence_MI_Dynamic_fixedEMA_relaxed
```

結果保存先:

```text
logs/case/compare_source_tent_oftta/260616_135905_CASE_arousal_MI_Dynamic_fixedEMA_relaxed/
logs/case/compare_source_tent_oftta/260616_135928_CASE_valence_MI_Dynamic_fixedEMA_relaxed/
```

CASE arousal:

| Method | Accuracy | Macro-F1 | Wins |
| --- | ---: | ---: | ---: |
| Source | 0.4959 | 0.4069 | 1 |
| Norm | 0.5674 | 0.5625 | 11 |
| Tent | 0.5673 | 0.5624 | 10 |
| OFTTA | 0.5663 | 0.5464 | 8 |
| EMA-Tent | 0.5337 | 0.4990 | 6 |
| MI gate only | 0.5204 | 0.4785 | 3 |
| MI dynamic fixed-EMA relaxed | 0.5180 | 0.4830 | 1 |

CASE valence:

| Method | Accuracy | Macro-F1 | Wins |
| --- | ---: | ---: | ---: |
| Source | 0.5348 | 0.4540 | 5 |
| Norm | 0.5868 | 0.5718 | 12 |
| Tent | 0.5868 | 0.5719 | 13 |
| OFTTA | 0.5868 | 0.5531 | 5 |
| EMA-Tent | 0.5608 | 0.5260 | 3 |
| MI gate only | 0.5450 | 0.5019 | 1 |
| MI dynamic fixed-EMA relaxed | 0.5492 | 0.5034 | 3 |

CASE では WESAD と異なり、Norm/Tent が最も高い平均 Macro-F1 を示した。
つまり CASE では、過去 target EMA や source/EMA/current の動的混合よりも、
現在 batch の BN 統計を直接使う単純な test-time normalization が有効である。

これは、CASE の target stream では WESAD ほど強いブロック構造・単一クラス batch 問題が
支配的ではない、または現在 batch 統計が十分に安定している可能性を示す。
したがって、MI-gated Dynamic EMA-Tent は WESAD の通常順序ブロック問題には有効だが、
CASE へそのまま汎用的に効く手法とはまだ言えない。

現時点での解釈:

```text
WESAD:
  block-like stream が強く、current batch 統計を制御しつつ EMA を維持する設計が有効。

CASE:
  current batch BN 統計が有効で、Norm/Tent が強い。
  MI gate による抑制はむしろ適応を弱める可能性がある。
```
