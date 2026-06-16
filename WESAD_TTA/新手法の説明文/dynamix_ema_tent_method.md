# DynaMix EMA-Tent

## 目的

DynaMix EMA-Tent は、既存の EMA-Tent を基準に、BatchNorm 統計の
`source / target EMA / current batch` の混合重みだけを動的に決める手法である。

既存 EMA-Tent では、混合重みを以下のような固定ハイパーパラメータとして設定していた。

```text
w_source = 0.2
w_ema    = 0.5
w_batch  = 0.3
```

しかし WESAD の通常順序では、target stream にブロック構造があり、batch 内ラベルが
単一クラスに大きく偏ることが多い。そのため、固定重みでは「どの程度 current batch
統計を信じるか」「どの程度 source 統計から target EMA 統計へ移行するか」を
手動で決める必要がある。

DynaMix EMA-Tent では、この固定混合比をやめ、target stream の進行に応じて
混合重みを動的に決定する。

## EMA-Tent との違い

DynaMix EMA-Tent は、EMA-Tent の以下の要素は維持する。

```text
1. BatchNorm affine パラメータ gamma, beta のみを entropy minimization で更新する
2. Conv / Linear 層は更新しない
3. source 統計、target EMA 統計、current batch 統計を混合してBN正規化する
4. target EMA 統計は固定 momentum で更新する
5. batch平均予測分布の entropy に基づく gate を使う
```

一方で、以下だけを変更する。

```text
EMA-Tent:
  source / EMA / current batch の混合重みを固定ハイパーパラメータで与える

DynaMix EMA-Tent:
  source / EMA / current batch の混合重みを batch ごとに動的に決める
```

したがって DynaMix EMA-Tent は、MI gate や gate-aware EMA を使わない。
元 EMA-Tent に対する主な差分は dynamic mixture weights のみである。

## Gate

DynaMix EMA-Tent では、MI gate ではなく、元 EMA-Tent と同じ考え方の
batch diversity gate を使う。

各サンプルの softmax 出力を `p_i`、batch 平均予測分布を `p_bar` とする。

```text
p_bar = (1 / B) sum_i p_i
```

batch 平均予測分布の entropy:

```text
H(p_bar) = - sum_c p_bar_c log p_bar_c
```

正規化した gate:

```text
g = H(p_bar) / log C
```

`g` が大きい場合、batch 全体の予測が複数クラスに分散していることを意味する。
`g` が小さい場合、予測が単一クラスに偏っており、current batch 統計を強く信じると
危険な可能性がある。

## 動的混合重み

DynaMix EMA-Tent では、current batch 統計の最大使用量を `w_batch_max` とする。
今回の設定では `w_batch_max = 0.3` を用いた。

```text
w_batch^t = w_batch_max * g_t
```

次に、target EMA 統計の信頼度 `rho_t` を、信頼できる target batch をどれだけ
観測してきたかに基づいて増加させる。

```text
N_eff^t = N_eff^(t-1) + B * g_t
rho_t = 1 - exp(-N_eff^t / K)
```

ここで `K` は EMA 統計へ移行する速さを決めるパラメータである。
今回の設定では `rho_k = 128` を用いた。

残りの重みを source と EMA に配分する。

```text
w_ema^t = (1 - w_batch^t) * rho_t
w_source^t = (1 - w_batch^t) * (1 - rho_t)
```

このため、DynaMix EMA-Tent は次の挙動になる。

```text
初期:
  source 統計を強めに使う

信頼できる target batch を観測した後:
  target EMA 統計を強めに使う

batch 予測が多様なとき:
  current batch 統計を一定量使う

batch 予測が単一クラス寄りのとき:
  current batch 統計の寄与を下げる
```

## EMA 更新

DynaMix EMA-Tent では、EMA 統計の更新率は gate で制御しない。
元 EMA-Tent と同様に固定 momentum を使う。

```text
mu_ema^t = m * mu_ema^(t-1) + (1 - m) * mu_batch^t
var_ema^t = m * var_ema^(t-1) + (1 - m) * var_batch^t
```

今回の設定では `m = 0.8` を用いた。

これは、WESAD の通常順序では単一クラス batch が多く、gate が小さくなりやすいためである。
EMA 更新まで gate で抑えると、target EMA 統計が十分に蓄積されず、source 統計へ
戻りすぎる傾向があった。

## 設定ファイル

```text
cfg/algorithm/dynamix_ema_tent.yaml
```

主要設定:

```yaml
adaption: mi_dynamic_ema_tent
tent_lr: 0.0001
tent_steps: 1
episodic: false

ema_tent_source_weight: 0.2
ema_tent_ema_weight: 0.5
ema_tent_batch_weight: 0.3
ema_momentum: 0.80

mi_gate: false
gate_aware_ema: false
dynamic_mixture_weights: true
gate_threshold: 0.2

w_batch_max: 0.3
rho_k: 128
```

実装上は `mi_dynamic_ema_tent` のクラスを再利用しているが、`mi_gate: false`、
`gate_aware_ema: false` としているため、MI gate や gate-aware EMA は使っていない。

## 実行例

WESAD 3class:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source tent oftta ema_tent dynamix_ema_tent \
  --run_name WESAD3class_TentOFTTAEMATent_DynaMixEMATent
```

CASE arousal:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_arousal.yaml \
  --resume ./ckpt_case_arousal \
  --methods source tent oftta ema_tent dynamix_ema_tent \
  --run_name CASE_arousal_SourceTentOFTTAEMATent_DynaMixEMATent
```

CASE valence:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_valence.yaml \
  --resume ./ckpt_case_valence \
  --methods source tent oftta ema_tent dynamix_ema_tent \
  --run_name CASE_valence_SourceTentOFTTAEMATent_DynaMixEMATent
```

## WESAD 結果

WESAD 3class 通常順序で、Source / Tent / OFTTA / EMA-Tent / DynaMix EMA-Tent を比較した。

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

DynaMix EMA-Tent は、ハイパーパラメータで固定混合重みを設定した EMA-Tent に対して、
平均 Macro-F1 で `+0.0036` 上回った。また、被験者別では 15 人中 10 人で
EMA-Tent より高い Macro-F1 を示した。

ただし改善幅は小さいため、「大幅に改善した」ではなく、以下のように解釈するのが妥当である。

```text
DynaMix EMA-Tent は、固定混合重みを用いる EMA-Tent と同等以上の性能を維持しつつ、
source / EMA / current batch の混合重みの手動設定への依存を軽減できる可能性を示した。
```

図:

```text
logs/wesad/compare_source_tent_oftta/260616_143909_WESAD3class_TentOFTTAEMATent_DynaMixEMATent/tent_oftta_ema_tent_dynamix_macro_f1.pdf
logs/wesad/compare_source_tent_oftta/260616_143909_WESAD3class_TentOFTTAEMATent_DynaMixEMATent/tent_oftta_ema_tent_dynamix_subject_macro_f1.pdf
```

## CASE 結果

CASE では arousal と valence の2タスクで評価した。

### CASE Arousal

結果保存先:

```text
logs/case/compare_source_tent_oftta/260616_144317_CASE_arousal_SourceTentOFTTAEMATent_DynaMixEMATent/
```

| Method | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Source | 0.4959 | 0.4069 |
| Tent | 0.5673 | 0.5624 |
| OFTTA | 0.5663 | 0.5464 |
| EMA-Tent | 0.5337 | 0.4990 |
| DynaMix EMA-Tent | 0.5250 | 0.4738 |

### CASE Valence

結果保存先:

```text
logs/case/compare_source_tent_oftta/260616_144411_CASE_valence_SourceTentOFTTAEMATent_DynaMixEMATent/
```

| Method | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Source | 0.5348 | 0.4540 |
| Tent | 0.5868 | 0.5719 |
| OFTTA | 0.5868 | 0.5531 |
| EMA-Tent | 0.5608 | 0.5260 |
| DynaMix EMA-Tent | 0.5649 | 0.5178 |

CASE では、WESAD と異なり Tent が最も高い平均 Macro-F1 を示した。
DynaMix EMA-Tent は EMA-Tent を上回る被験者もあるが、平均では EMA-Tent より低い。

この結果から、DynaMix EMA-Tent は CASE 全般に汎用的に効くというより、
WESAD の通常順序に含まれるブロック構造・単一クラス batch 問題に対して有効な
可能性が高い。

## 論文での位置付け

DynaMix EMA-Tent は、以下のように位置付けるのがよい。

```text
EMA-Tentでは、source / EMA / current batch の混合重みを固定ハイパーパラメータとして
設定する必要があった。DynaMix EMA-Tentでは、target streamの進行とbatch予測分布の
多様性に基づいて混合重みを動的に決定する。WESADでは、固定混合重みを用いるEMA-Tentと
同等以上のMacro-F1を維持しつつ、混合重みの手動設定への依存を軽減できる可能性を示した。
```

注意点として、DynaMix EMA-Tent は完全にハイパーパラメータフリーではない。
`w_batch_max`、`rho_k`、`ema_momentum` などは残る。ただし、EMA-Tent の中心的な
固定混合比である `w_source / w_ema / w_batch` を直接指定する必要は弱くなる。

