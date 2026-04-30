# EMA-Tent safe/default/adaptive 比較

## 比較対象

通常順序条件で，EMA-Tent の基本補間比率を変えた3条件を比較した。

| Variant | Source weight | EMA weight | Batch weight | Momentum | Log |
|---|---:|---:|---:|---:|---|
| safe | 0.5 | 0.4 | 0.1 | 0.9 | `260420_192036_EMATent_safe` |
| default | 0.3 | 0.5 | 0.2 | 0.9 | `260420_192526_EMATent_default` |
| adaptive | 0.2 | 0.5 | 0.3 | 0.85 | `260420_192938_EMATent_adaptive` |

ここで，これらの値は固定の基本補間比率である。ただし，実際の current batch 統計の寄与は batch 予測分布の多様性 gate により動的に調整される。

## 平均性能

| Variant | Accuracy | F1 baseline | F1 stress | F1 amusement | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| safe | 0.7000 | 0.8753 | 0.5499 | 0.2777 | 0.5676 |
| default | 0.7090 | 0.8725 | 0.5785 | 0.3213 | 0.5908 |
| adaptive | 0.7109 | 0.8543 | 0.6100 | 0.3677 | 0.6107 |

平均Macro-F1では adaptive が最も高い。default から adaptive への改善は +0.0199，safe から adaptive への改善は +0.0431 であった。

adaptive は baseline F1 をやや下げる一方で，stress F1 と amusement F1 を改善している。特に amusement F1 は safe 0.2777，default 0.3213，adaptive 0.3677 と増加しており，current batch 統計の寄与を強めたことが少数・変動クラスの適応に効いた可能性がある。

## 被験者別の勝敗

| Variant | Best subjects |
|---|---:|
| safe | 7 / 15 |
| default | 1 / 15 |
| adaptive | 7 / 15 |

被験者別では adaptive が全被験者で一貫して最良というわけではない。safe と adaptive がそれぞれ7被験者で最良であり，default が1被験者で最良であった。

adaptive が default を上回った被験者は8/15，下回った被験者は7/15である。したがって，adaptive の平均性能が高い理由は，全体で一様に改善したというより，一部被験者で大きく改善した影響がある。

特に S16 では adaptive - default が +0.2304 と大きく，平均差を押し上げている。一方，S15 では adaptive - default が -0.0485，S11では -0.0305 であり，adaptive が悪化する被験者も存在する。

## Paired comparison

被験者ごとのMacro-F1差分で比較した。

| Comparison | Mean diff | Median diff | Positive subjects | Negative subjects |
|---|---:|---:|---:|---:|
| default - safe | +0.0231 | +0.0029 | 8 | 7 |
| adaptive - default | +0.0199 | +0.0000 | 8 | 7 |
| adaptive - safe | +0.0431 | -0.0013 | 7 | 8 |

平均差では adaptive が最も良いが，中央値では adaptive - default はほぼ0であり，adaptive - safe はわずかに負である。このため，現時点で言えるのは「平均Macro-F1ではadaptiveが最良」であり，「adaptiveが全被験者に対して一貫して最良」とまでは言えない。

## どのハイパーパラメータが効いたか

この3条件だけでは，どのハイパーパラメータが最適かを一意に判断できない。理由は，safe/default/adaptiveで複数の値を同時に変更しているためである。

| 変更 | safe -> default | default -> adaptive |
|---|---|---|
| Source weight | 0.5 -> 0.3 | 0.3 -> 0.2 |
| EMA weight | 0.4 -> 0.5 | 0.5 -> 0.5 |
| Batch weight | 0.1 -> 0.2 | 0.2 -> 0.3 |
| Momentum | 0.9 -> 0.9 | 0.9 -> 0.85 |

defaultからadaptiveでは，batch weightを0.2から0.3へ増やし，source weightを0.3から0.2へ下げ，momentumを0.9から0.85へ下げている。そのため，改善が以下のどれによるものかは切り分けられていない。

- current batch 統計を強めた効果
- source anchor を弱めた効果
- EMA momentum を下げてtarget統計へ速く追従した効果
- これらの相互作用

## 現時点の解釈

平均Macro-F1だけを見るなら adaptive が最良である。特に stress と amusement のF1が改善しているため，WESADの通常順序条件では，source anchorを少し弱め，current batch統計とtarget EMAへの追従を強める方向が有効である可能性が高い。

ただし，被験者別の勝敗は safe と adaptive で割れている。したがって，論文では次のように書くのが安全である。

```text
Among the tested EMA-Tent variants, the adaptive setting achieved the highest average Macro-F1. However, because multiple hyperparameters were changed simultaneously and the best variant differed across subjects, further controlled ablations are required to identify the individual contribution of each hyperparameter.
```

日本語では以下のように書ける。

```text
検討した3条件の中では，adaptive設定が最も高い平均Macro-F1を示した。一方で，被験者別にはsafeとadaptiveで最良条件が分かれており，adaptiveが全被験者で一貫して優れるわけではなかった。また，defaultからadaptiveではsource weight，batch weight，momentumを同時に変更しているため，どのハイパーパラメータが性能改善に寄与したかはこの比較だけでは同定できない。したがって，最適なハイパーパラメータを決定するには，1要因ずつ変更する追加アブレーションが必要である。
```

## 次に行うべき探索

どのハイパーパラメータが効いているかを調べるには，以下の2段階で切り分けるのがよい。

### 1. Weight sweep

momentumを0.85に固定し，source weightとbatch weightのトレードオフを見る。

| Setting | Source | EMA | Batch | Momentum |
|---|---:|---:|---:|---:|
| more adaptive | 0.1 | 0.5 | 0.4 | 0.85 |
| current best | 0.2 | 0.5 | 0.3 | 0.85 |
| default-like | 0.3 | 0.5 | 0.2 | 0.85 |
| safer | 0.4 | 0.5 | 0.1 | 0.85 |

これにより，current batch統計を強めるほど性能が上がるのか，または0.3付近で頭打ちになるのかを確認できる。

### 2. Momentum sweep

source/EMA/batchを0.2/0.5/0.3に固定し，momentumだけを変える。

| Setting | Source | EMA | Batch | Momentum |
|---|---:|---:|---:|---:|
| fast EMA | 0.2 | 0.5 | 0.3 | 0.75 |
|  | 0.2 | 0.5 | 0.3 | 0.80 |
| current best | 0.2 | 0.5 | 0.3 | 0.85 |
|  | 0.2 | 0.5 | 0.3 | 0.90 |
| slow EMA | 0.2 | 0.5 | 0.3 | 0.95 |

これにより，adaptiveの改善がmomentum 0.85によるものか，重み設定によるものかを切り分けられる。

上記の探索用スクリプトは以下に追加した。

```text
scripts/wesad/compare_ema_tent_hparam_grid_wesad_3class.sh
```

