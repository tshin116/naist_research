# EMA-Tent

## 目的

`ema_tent` は、WESAD の時系列ブロック構造により通常 Tent が不安定になる問題を緩和するための Tent 派生手法である。

通常 Tent では、BatchNorm1d の running statistics を使わず、現在の test batch の統計だけで正規化する。WESAD の target loader を時系列順に評価すると、同一状態の window が連続しやすく、batch が特定クラスや特定状態に偏りやすい。そのため、現在 batch の統計だけを使うと、モデルが一時的なブロックに過剰適応しやすい。

EMA-Tent では、BatchNorm 統計を現在 batch のみに依存させず、source running statistics と過去 test batch の EMA 統計を併用する。

ここで3成分に分ける理由は、それぞれが異なる役割を持つためである。

- **source running statistics**:
  学習時分布に基づく最も保守的な anchor。偏った test batch が来ても完全には崩れない基準になる。
- **past test EMA statistics**:
  target 被験者を処理する中で蓄積される長期的な target 情報。単一 batch よりも target 個人全体の傾向を表しやすい。
- **current batch statistics**:
  現在の状態変化に追従する局所適応成分。短期的な状態遷移をすばやく反映できる。

特に、source と EMA を分けることが重要である。EMA は source 統計で初期化されるが、test stream を処理するにつれて target 個人の統計へ徐々に移動していく。もし source と EMA をまとめてしまうと、「学習時分布への安全な anchor」と「target 個人へ適応した履歴」を区別できず、初期の偏った batch や過去状態に引きずられたときに戻る基準が弱くなる。

## 通常 Tent との違い

通常 Tent は以下を行う。

1. BatchNorm1d を test batch statistics で動かす
2. 予測 entropy を最小化する
3. BatchNorm1d の affine parameter、すなわち `weight` と `bias` だけを更新する

EMA-Tent はこの方針を維持しつつ、BatchNorm 統計を次の3成分の混合に変更する。

```text
source running statistics
+ past test EMA statistics
+ current batch statistics
```

また、現在 batch の予測分布が単一クラスへ偏っている場合は、current batch statistics の寄与と entropy minimization 更新を弱める。

## 処理の流れ

1 batch の forward では、以下の処理を行う。

```text
input batch
  ↓
source/EMA寄りのBN統計で probe forward
  ↓
batch予測分布の多様性から gate を計算
  ↓
gate に応じて current batch statistics の寄与を調整
  ↓
source + EMA + current batch 統計で adapt forward
  ↓
現在 batch の logits を評価出力として返す
  ↓
gate が十分大きい場合のみ entropy loss で BN affine を更新
  ↓
現在 batch の統計で EMA を更新
```

評価に使う logits は、既存 Tent と同じく、その batch の entropy update 前の forward 出力である。更新は次 batch 以降に反映される。

## EMA BatchNorm

通常 Tent の BatchNorm 統計は次のように表せる。

```text
mean = current_batch_mean
std  = current_batch_std
```

EMA-Tent では次のように混合する。

```text
mean = w_source * source_mean
     + w_ema    * ema_test_mean
     + w_batch  * current_batch_mean

std  = w_source * source_std
     + w_ema    * ema_test_std
     + w_batch  * current_batch_std
```

`ema_test_mean` と `ema_test_std` は、初期値として source running statistics を持ち、test batch を処理するたびに更新される。

```text
ema_mean = momentum * ema_mean + (1 - momentum) * current_batch_mean
ema_std  = momentum * ema_std  + (1 - momentum) * current_batch_std
```

これにより、現在 batch の局所統計だけでなく、過去に観測した target 被験者の統計を蓄積して利用できる。

## Batch Bias Gate

WESAD の時系列順 batch は単一状態へ偏る可能性があるため、EMA-Tent は batch の予測分布から gate を計算する。

```text
pred_dist = mean softmax probability over batch
diversity = normalized entropy(pred_dist)
gate = diversity ^ gate_power
```

`gate` は 0 から 1 の値である。

| gate | 意味 |
|---|---|
| 1 に近い | batch の予測分布が多様で、current batch statistics を比較的信頼できる |
| 0 に近い | batch が単一クラスへ偏っており、current batch statistics を信頼しにくい |

直感的には、`gate` は「今見ている batch をどれだけ信用してよいか」を表す。複数クラスが混ざった batch なら、その batch は target 分布の一部をある程度代表していると考えられるため、`gate` は大きくなる。逆に、予測がほぼ1クラスに集中している batch は、WESAD のブロック構造に由来する単一状態区間を見ている可能性が高く、target 分布全体を代表していない。このとき `gate` は小さくなる。

したがって、`gate` が小さいとは「現在 batch の統計をそのまま強く使うのは危険」という意味である。EMA-Tent はこの状況で current batch statistics の重みを下げ、source statistics と EMA statistics をより強く使う。これにより、単一クラスのブロック区間で BN 統計がその区間に一気に引きずられることを防ぐ。

`gate` が小さい場合、current batch statistics の重みを source 側へ戻す。

```text
w_batch_effective = w_batch * gate
w_source_effective = w_source + w_batch * (1 - gate)
```

これにより、偏った batch では source statistics と EMA statistics の影響が強くなる。

## Entropy Update Gate

EMA-Tent では、`gate` が `ema_tent_gate_threshold` 未満の場合、entropy minimization による BN affine 更新を行わない。

```text
if gate < ema_tent_gate_threshold:
    skip entropy update
```

また、`ema_tent_loss_gate: true` の場合は、entropy loss に gate を掛ける。

```text
loss = gate * entropy_loss
```

これにより、batch が強く偏っているときにBN affine parameterが大きく動くことを抑える。

## 設定値

設定は `cfg/algorithm/ema_tent.yaml` にある。

| key | 意味 | 主設定 |
|---|---|---:|
| `tent_lr` | BN affine 更新の学習率 | 0.0001 |
| `tent_steps` | 1 batch あたりの更新回数 | 1 |
| `episodic` | 各 batch 前に初期状態へ戻すか | false |
| `ema_tent_source_weight` | source statistics の基本重み | 0.2 |
| `ema_tent_ema_weight` | test EMA statistics の基本重み | 0.5 |
| `ema_tent_batch_weight` | current batch statistics の基本重み | 0.3 |
| `ema_tent_momentum` | test statistics EMA の momentum | 0.8 |
| `ema_tent_use_gate` | batch bias gate を使うか | true |
| `ema_tent_gate_threshold` | entropy update を行う gate しきい値 | 0.2 |
| `ema_tent_gate_power` | gate の強さを調整する指数 | 1.0 |
| `ema_tent_min_confidence` | entropy update に必要な平均confidence | 0.0 |
| `ema_tent_loss_gate` | entropy loss に gate を掛けるか | true |
| `ema_tent_probe_gate` | gate計算用probe forward時のBN gate | 0.0 |

主設定は 260421 の hyperparameter grid search に基づき選択した。grid上の平均Macro-F1最良は `source=0.1, ema=0.5, batch=0.4, momentum=0.85` であったが，`source=0.2, ema=0.5, batch=0.3, momentum=0.80` との差は +0.0007 Macro-F1 と小さかった。そのため，source anchorを残す設計意図との整合性も考慮し，後者を主設定として用いる。

## 実行方法

特定被験者だけ評価する場合:

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env python adapt.py \
  --target_domain S2 \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --algorithm_cfg ./cfg/algorithm/ema_tent.yaml \
  --resume ./ckpt_3class
```

全被験者で EMA-Tent だけ評価する場合:

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
bash scripts/wesad/adapt_ema_tent_wesad_3class.sh
```

Source、Tent、EMA-Tent、OFTTA をまとめて比較する場合:

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
bash scripts/wesad/compare_source_tent_ema_tent_oftta_wesad_3class.sh
```

shuffle条件で比較する場合:

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
bash scripts/wesad/compare_source_tent_ema_tent_oftta_shuffle_wesad_3class.sh
```

## 実験結果の要点

通常順序の WESAD 3分類評価では、EMA-Tent は Source、Tent、OFTTA を平均性能で上回った。

ここで Macro-F1 は、baseline、stress、amusement の各クラスF1-scoreを単純平均した値である。

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Source | 0.6915 | 0.5285 |
| Tent | 0.4553 | 0.3970 |
| EMA-Tent | 0.7075 | 0.6136 |
| OFTTA | 0.5231 | 0.4499 |

EMA-Tent は特に Stress と Amusement の F1 を改善した。

| Metric | EMA-Tent - Source |
|---|---:|
| F1 Stress | +0.1477 |
| F1 Amusement | +0.1454 |

この結果は、current batch statistics のみに依存する通常 Tent よりも、過去 test statistics を EMA として蓄積する方が、WESAD のブロック構造に対して安定であることを示している。

## 実装ファイル

- `TTA/adapt_algorithm/ema_tent.py`
- `cfg/algorithm/ema_tent.yaml`
- `scripts/wesad/adapt_ema_tent_wesad_3class.sh`
- `scripts/wesad/compare_source_tent_ema_tent_oftta_wesad_3class.sh`
- `scripts/wesad/compare_source_tent_ema_tent_oftta_shuffle_wesad_3class.sh`
