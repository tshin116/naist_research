# EMA-Tent hyperparameter grid 分析

## 対象ログ

`scripts/wesad/compare_ema_tent_hparam_grid_wesad_3class.sh` により実行された，2026-04-21の通常順序評価ログを分析した。

| Log | Variant | Source | EMA | Batch | Momentum |
|---|---|---:|---:|---:|---:|
| `260421_230849` | `s10_e50_b40_m085` | 0.1 | 0.5 | 0.4 | 0.85 |
| `260421_231351` | `s20_e50_b30_m085` | 0.2 | 0.5 | 0.3 | 0.85 |
| `260421_231819` | `s30_e50_b20_m085` | 0.3 | 0.5 | 0.2 | 0.85 |
| `260421_232238` | `s40_e50_b10_m085` | 0.4 | 0.5 | 0.1 | 0.85 |
| `260421_232657` | `s20_e50_b30_m075` | 0.2 | 0.5 | 0.3 | 0.75 |
| `260421_233138` | `s20_e50_b30_m080` | 0.2 | 0.5 | 0.3 | 0.80 |
| `260421_233559` | `s20_e50_b30_m090` | 0.2 | 0.5 | 0.3 | 0.90 |
| `260421_234019` | `s20_e50_b30_m095` | 0.2 | 0.5 | 0.3 | 0.95 |

## 平均性能ランキング

| Rank | Variant | Accuracy | F1 baseline | F1 stress | F1 amusement | Macro-F1 | Std Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `s10_e50_b40_m085` | 0.7141 | 0.8568 | 0.6174 | 0.3686 | 0.6142 | 0.1542 |
| 2 | `s20_e50_b30_m080` | 0.7075 | 0.8413 | 0.6224 | 0.3771 | 0.6136 | 0.1655 |
| 3 | `s20_e50_b30_m075` | 0.6987 | 0.8245 | 0.6297 | 0.3782 | 0.6108 | 0.1676 |
| 4 | `s20_e50_b30_m085` | 0.7109 | 0.8543 | 0.6100 | 0.3677 | 0.6107 | 0.1593 |
| 5 | `s40_e50_b10_m085` | 0.7102 | 0.8558 | 0.6015 | 0.3725 | 0.6099 | 0.1673 |
| 6 | `s30_e50_b20_m085` | 0.7092 | 0.8537 | 0.6036 | 0.3697 | 0.6090 | 0.1621 |
| 7 | `s20_e50_b30_m090` | 0.7104 | 0.8719 | 0.5846 | 0.3237 | 0.5934 | 0.1340 |
| 8 | `s20_e50_b30_m095` | 0.7015 | 0.8804 | 0.5442 | 0.2666 | 0.5637 | 0.1274 |

平均Macro-F1では `s10_e50_b40_m085` が最良であった。これは，source anchorを最も弱くし，current batchの基本重みを最も強くした設定である。

ただし，1位の `s10_e50_b40_m085` と2位の `s20_e50_b30_m080` の差は +0.0007 Macro-F1 と非常に小さい。上位4条件は 0.6107から0.6142の範囲に収まっており，明確に1条件だけが突出しているわけではない。

## Weight sweepの解釈

momentumを0.85に固定し，EMA weightを0.5に固定したまま，source weightとbatch weightを変えた結果は以下である。

| Source | EMA | Batch | Momentum | Macro-F1 |
|---:|---:|---:|---:|---:|
| 0.1 | 0.5 | 0.4 | 0.85 | 0.6142 |
| 0.2 | 0.5 | 0.3 | 0.85 | 0.6107 |
| 0.3 | 0.5 | 0.2 | 0.85 | 0.6090 |
| 0.4 | 0.5 | 0.1 | 0.85 | 0.6099 |

この範囲では，source weightを弱め，batch weightを強めた `0.1/0.5/0.4` が最良であった。したがって，WESADの通常順序評価では，current batch統計をある程度強く使う方向が有効である可能性がある。

ただし，`0.2/0.5/0.3`，`0.3/0.5/0.2`，`0.4/0.5/0.1` の差は小さい。また，current batch weightをさらに増やした場合に性能が上がるか，過適応するかは未確認である。

## Momentum sweepの解釈

source/EMA/batch weightを0.2/0.5/0.3に固定し，momentumのみを変えた結果は以下である。

| Source | EMA | Batch | Momentum | Macro-F1 |
|---:|---:|---:|---:|---:|
| 0.2 | 0.5 | 0.3 | 0.75 | 0.6108 |
| 0.2 | 0.5 | 0.3 | 0.80 | 0.6136 |
| 0.2 | 0.5 | 0.3 | 0.85 | 0.6107 |
| 0.2 | 0.5 | 0.3 | 0.90 | 0.5934 |
| 0.2 | 0.5 | 0.3 | 0.95 | 0.5637 |

momentumは0.80が最良であり，0.75から0.85は近い性能を示した。一方，0.90以上ではMacro-F1が大きく低下した。

これは，WESADのtarget streamでは過去test統計を蓄積しつつも，あまり遅いEMAではtarget個人や状態遷移に追従できない可能性を示す。特にmomentum 0.95ではbaseline F1は高いが，stress F1とamusement F1が大きく下がっており，source寄り・過去統計寄りに残りすぎてtarget状態への適応が不足したと考えられる。

## 被験者別の最良設定

| Variant | Best subjects |
|---|---:|
| `s20_e50_b30_m095` | 6 |
| `s20_e50_b30_m075` | 5 |
| `s10_e50_b40_m085` | 1 |
| `s30_e50_b20_m085` | 1 |
| `s20_e50_b30_m080` | 1 |
| `s20_e50_b30_m090` | 1 |
| `s20_e50_b30_m085` | 0 |
| `s40_e50_b10_m085` | 0 |

被験者別の最良設定は大きくばらついた。平均性能が最良の `s10_e50_b40_m085` は，被験者別の勝利数では1/15のみであった。一方，`s20_e50_b30_m095` は6/15被験者で最良であるが，平均Macro-F1は最下位であった。

これは，`m095` が一部被験者では安定する一方で，他の被験者でstress/amusement F1を大きく落とすため，平均性能が低くなったことを示す。論文では，被験者別最良数よりも，平均Macro-F1とクラス別F1を主指標にした方がよい。

## 既存手法との比較

平均Macro-F1最良の `s10_e50_b40_m085` は，既存手法に対して以下の改善を示した。

| Comparison | Mean diff | Median diff | Improved subjects | Worsened subjects |
|---|---:|---:|---:|---:|
| `s10_e50_b40_m085` - Source | +0.0858 | +0.0541 | 10 | 5 |
| `s10_e50_b40_m085` - Tent | +0.2172 | +0.1547 | 15 | 0 |
| `s10_e50_b40_m085` - OFTTA | +0.1643 | +0.1744 | 13 | 2 |

`s10_e50_b40_m085` はTentに対して全被験者で改善し，OFTTAに対しても13/15被験者で改善した。Sourceに対しては10/15被験者で改善した。

## 上位条件間の差

上位条件間のpaired差分は小さい。

| Comparison | Mean diff | Median diff | Positive | Negative | paired t-test p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| `s10_e50_b40_m085` - `s20_e50_b30_m080` | +0.0007 | +0.0025 | 8 | 7 | 0.9085 | 0.5245 |
| `s10_e50_b40_m085` - `s20_e50_b30_m075` | +0.0034 | +0.0106 | 10 | 5 | 0.7322 | 0.4887 |
| `s10_e50_b40_m085` - `s20_e50_b30_m085` | +0.0036 | +0.0022 | 10 | 5 | 0.2835 | 0.1688 |

これらの差は統計的に明確ではない。したがって，現時点では `s10_e50_b40_m085` を「平均Macro-F1が最も高い候補」として扱い，「有意に最良」とは書かない方がよい。

## 結論

今回のgridでは，平均Macro-F1の最良設定は以下であった。

```text
source = 0.1
ema = 0.5
batch = 0.4
momentum = 0.85
Macro-F1 = 0.6142
```

一方，momentum sweepでは `source/ema/batch = 0.2/0.5/0.3` のもとで `momentum = 0.80` が最良であり，Macro-F1は0.6136であった。両者の差は+0.0007しかなく，実質的には同等である。

論文・今後の実験では，次のどちらかを採用するのが妥当である。

1. **平均性能重視**: `source=0.1, ema=0.5, batch=0.4, momentum=0.85`
2. **安定した解釈重視**: `source=0.2, ema=0.5, batch=0.3, momentum=0.80`

前者は今回のgridで平均Macro-F1が最も高い。後者はadaptive設定に近く，momentum sweep上で最良であり，手法説明としても「source anchorを残しつつ，target EMAとcurrent batchをやや強める」と説明しやすい。

現時点でのハイパーパラメータに関する主張は以下が安全である。

```text
The grid search suggests that EMA-Tent benefits from relatively fast target-statistics adaptation and a larger current-batch contribution than the default setting. However, the top configurations show very similar Macro-F1, and the best subject-wise configuration varies across subjects. Therefore, we use the grid result to select a strong configuration, but do not claim a universally optimal hyperparameter setting.
```

日本語では以下のように書ける。

```text
今回のgrid searchでは，default設定よりもsource統計の重みを弱め，current batch統計の重みを強め，target EMAを比較的速く追従させる設定が高い平均Macro-F1を示した。ただし，上位設定間の差は小さく，被験者ごとの最良設定も一致しなかった。そのため，本結果は強い候補設定を選ぶための探索結果として扱い，普遍的な最適ハイパーパラメータを同定したとは主張しない。
```

