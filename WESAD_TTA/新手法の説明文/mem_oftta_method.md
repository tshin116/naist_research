# Memory-Balanced OFTTA

## 目的

`mem_oftta` は、WESAD の時系列ブロック構造によって OFTTA / Tent が batch composition に強く依存する問題を緩和するための実験的手法である。

WESAD の target loader を時系列順に評価すると、各 batch が Baseline、Stress、Amusement のいずれかに強く偏りやすい。このような batch に対して通常の Test-Time Adaptation を行うと、現在 batch の局所分布に過剰適応し、後続 batch の性能を悪化させる可能性がある。`mem_oftta` は、現在 batch のみに依存せず、source prototype と class-balanced memory を使って分類器調整を行う。

## 既存 OFTTA との違い

既存の `oftta` は、主に次の2つを行う。

1. BatchNorm の source running statistics と test batch statistics を重み付きで混合する
2. T3A と同様に、test feature と疑似ラベルから分類器を動的に作る

`mem_oftta` はこの方針を維持しつつ、以下を追加する。

- source classifier weight を常に prototype anchor として残す
- target feature を class-balanced memory に蓄積する
- long memory と short memory を分ける
- batch がクラス偏りしている場合、test batch statistics の混合を弱める
- batch がクラス偏りまたは低信頼の場合、support memory 更新を抑制する
- feature 分布または予測分布が急変した場合、short memory を reset する
- 勾配更新は行わず、model weight は変更しない

## 処理の流れ

1 batch の forward では、以下の順に処理する。

```text
input batch
  ↓
source-anchored BN で probe forward
  ↓
予測分布の偏り・信頼度から gate を計算
  ↓
feature 分布 / 予測分布の急変を検出
  ↓
必要なら short memory を reset
  ↓
gate に応じて Weighted BN の test 統計混合を調整
  ↓
adapt forward
  ↓
高信頼 sample だけ memory に追加
  ↓
source anchor + long memory + short memory から class-balanced support を作成
  ↓
prototype classifier で logits を出力
```

## Source Prototype Anchor

`mem_oftta` は最終線形分類器の重みを class prototype として保存し、常に support set に含める。

```text
anchor_supports = classifier.weight
```

二値分類でモデルが single-logit を返す場合は、既存の共通関数により 2 クラス用の重みに変換して扱う。3分類では分類器の3クラス重みをそのまま使う。

この anchor は、target memory が誤った疑似ラベルで汚染された場合でも、分類器が source model から完全に離れないようにするための基準である。

## Class-Balanced Memory

target feature は疑似ラベルと entropy とともに memory に保存される。ただし、全サンプルを保存するのではなく、平均信頼度がしきい値以上のサンプルだけを保存する。

```text
confidence = max softmax probability
confidence >= mem_oftta_confidence_threshold のサンプルだけ保存
```

memory は次の2種類に分かれる。

| memory | 役割 |
|---|---|
| long memory | target 被験者全体の安定した support を保持する |
| short memory | 現在の近傍ブロックに対応する support を保持する |

各 memory はクラスごとに低 entropy サンプルを最大 K 個だけ残す。

```text
long memory:  各クラス mem_oftta_long_K 個
short memory: 各クラス mem_oftta_short_K 個
```

これにより、時系列順 batch で一時的に同一クラスばかり来ても、support set 全体が単一クラスへ過度に偏ることを抑える。

## Batch Bias Gate

`mem_oftta` では、まず source-anchored BN に近い状態で probe forward を行い、batch の予測分布を確認する。

```text
pred_dist = mean softmax probability over batch
diversity = normalized entropy(pred_dist)
confidence = mean max softmax probability
gate = diversity ^ gate_power * confidence
```

`gate` は 0 から 1 の値である。

| gate | 意味 |
|---|---|
| 1 に近い | batch の予測分布が多様で、比較的信頼できる |
| 0 に近い | batch が単一クラスへ偏っている、または低信頼 |

この gate は2つの用途で使う。

1. Weighted BN の test batch statistics 混合を調整する
2. target support memory を更新するかどうかを決める

## Gated Weighted BN

通常の OFTTA では、各 BatchNorm 層で source 統計と test batch 統計を固定 prior で混合する。

```text
mean = prior * source_mean + (1 - prior) * batch_mean
std  = prior * source_std  + (1 - prior) * batch_std
```

`mem_oftta` では、batch gate により test batch statistics の影響を弱める。

```text
effective_prior = 1 - gate * (1 - base_prior)
mean = effective_prior * source_mean + (1 - effective_prior) * batch_mean
std  = effective_prior * source_std  + (1 - effective_prior) * batch_std
```

したがって、偏った batch では `gate` が小さくなり、`effective_prior` は 1 に近づく。この場合、BatchNorm は source 統計をより強く使う。

## Memory Update Gate

`gate` が `mem_oftta_update_gate_threshold` 未満の場合、その batch から target memory への support 追加を行わない。

```text
if gate < mem_oftta_update_gate_threshold:
    skip memory update
```

これは、単一状態に偏った batch や、予測が不安定な batch から誤った疑似ラベルを memory に入れることを防ぐためである。

## Block Transition Detection

WESAD では Baseline、Stress、Amusement がブロックとして出現しやすい。そのため、直前 batch と現在 batch の分布が大きく変わった場合、short memory を reset する。

`mem_oftta` では次の2つを見ている。

- normalized feature mean の L2 distance
- 予測分布の Jensen-Shannon divergence

どちらかがしきい値を超えた場合、block transition とみなし short memory を reset する。

```text
feature_shift > mem_oftta_feature_shift_threshold
or
pred_shift > mem_oftta_pred_shift_threshold
```

long memory と source anchor は reset しない。これにより、target 全体の安定した情報は残しつつ、直前ブロックに過剰適応した短期情報だけを捨てる。

## 出力分類器

最終的な分類器は、以下の support から作る。

```text
source anchor
+ long memory
+ short memory
```

support はクラスごとに低 entropy サンプルを選び、L2 normalize した上で prototype classifier を構成する。

```text
weights = normalized_supports.T @ labels
logits = feature @ normalize(weights)
```

これは T3A / OFTTA と同じ方向の classifier adjustment である。ただし、support set が現在 batch だけに引っ張られないよう、anchor と memory を明示的に使う点が異なる。

## 設定値

設定は `cfg/algorithm/mem_oftta.yaml` にある。

| key | 意味 | 初期値 |
|---|---|---:|
| `filter_K` | 最終support選択で各クラスに残す最大数 | 16 |
| `oftta_prior_min` | 前段BNの最小source prior | 0.1 |
| `oftta_prior_max` | 後段BNの最大source prior | 0.99 |
| `mem_oftta_long_K` | long memoryで各クラスに残す最大数 | 32 |
| `mem_oftta_short_K` | short memoryで各クラスに残す最大数 | 16 |
| `mem_oftta_confidence_threshold` | memory保存に必要なconfidenceしきい値 | 0.75 |
| `mem_oftta_update_gate_threshold` | memory更新に必要なgateしきい値 | 0.35 |
| `mem_oftta_gate_power` | gateの強さを調整する指数 | 1.0 |
| `mem_oftta_probe_gate` | probe forward時のBN gate | 0.0 |
| `mem_oftta_feature_shift_threshold` | short memory reset用feature shiftしきい値 | 0.65 |
| `mem_oftta_pred_shift_threshold` | short memory reset用prediction shiftしきい値 | 0.35 |

## 実行方法

特定被験者だけ評価する場合:

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env python adapt.py \
  --target_domain S2 \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --algorithm_cfg ./cfg/algorithm/mem_oftta.yaml \
  --resume ./ckpt_3class
```

全被験者で `mem_oftta` だけ評価する場合:

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
bash scripts/wesad/adapt_mem_oftta_wesad_3class.sh
```

Source、Tent、OFTTA、MemOFTTA をまとめて比較する場合:

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
bash scripts/wesad/compare_source_tent_oftta_mem_oftta_wesad_3class.sh
```

shuffle条件で比較する場合:

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
bash scripts/wesad/compare_source_tent_oftta_mem_oftta_shuffle_wesad_3class.sh
```

## 実験上の注意

`mem_oftta` は安全側の初期設定になっている。特に `mem_oftta_confidence_threshold` と `mem_oftta_update_gate_threshold` が高すぎると、target memory がほとんど更新されず、Source に近い挙動になる。

性能改善を狙う場合、最初に調整する候補は以下である。

```yaml
mem_oftta_confidence_threshold: 0.60
mem_oftta_update_gate_threshold: 0.20
```

ただし、しきい値を下げすぎると誤った疑似ラベルが memory に入り、OFTTA と同様に batch composition の影響を受けやすくなる。通常順序評価では、Sourceからの悪化を抑えつつ、shuffle条件の改善に近づけることを目標に調整する。

## 実装ファイル

- `TTA/adapt_algorithm/mem_oftta.py`
- `cfg/algorithm/mem_oftta.yaml`
- `scripts/wesad/adapt_mem_oftta_wesad_3class.sh`
- `scripts/wesad/compare_source_tent_oftta_mem_oftta_wesad_3class.sh`
- `scripts/wesad/compare_source_tent_oftta_mem_oftta_shuffle_wesad_3class.sh`
