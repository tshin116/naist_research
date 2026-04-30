# EMA-Tent 導入後評価の分析

## 概要

本分析では、WESAD 3分類設定において、通常順序の target loader で Source、Tent、EMA-Tent、OFTTA を比較した。評価対象ログは以下である。

```text
logs/wesad/compare_source_tent_oftta/260420_151858_EMATent手法導入後評価
```

この評価は target 被験者の前処理済み window を保存順に batch 化している。したがって、WESAD のブロック構造、すなわち同一状態に由来する window が連続しやすい条件での性能比較である。

## 平均性能

15被験者の平均性能を以下に示す。

| Method | Accuracy | Macro-F1 | F1 class 0 | F1 class 1 | F1 class 2 |
|---|---:|---:|---:|---:|---:|
| Source | 0.6915 | 0.5285 | 0.8790 | 0.4747 | 0.2317 |
| Tent | 0.4553 | 0.3970 | 0.5461 | 0.4340 | 0.2110 |
| EMA-Tent | 0.7090 | 0.5908 | 0.8725 | 0.5785 | 0.3213 |
| OFTTA | 0.5231 | 0.4499 | 0.6421 | 0.4629 | 0.2447 |

EMA-Tent は Accuracy と Macro-F1 の両方で最も高い平均性能を示した。Macro-F1 は Source の 0.5285 から 0.5908 へ改善し、通常 Tent の 0.3970 および OFTTA の 0.4499 を大きく上回った。

## Source に対する改善

EMA-Tent と Source の平均差分を以下に示す。

| Metric | EMA-Tent - Source |
|---|---:|
| Accuracy | +0.0175 |
| Macro-F1 | +0.0623 |
| F1 class 0 | -0.0065 |
| F1 class 1 | +0.1038 |
| F1 class 2 | +0.0896 |

EMA-Tent は class 0、すなわち Baseline の F1 をほぼ維持しつつ、Stress と Amusement の F1 を改善した。Source は Baseline に強く偏る傾向があり、class 1 と class 2 の検出が弱かった。EMA-Tent は BatchNorm 統計を source running stats、過去 test EMA stats、現在 batch stats の混合にすることで、Source の安定性を残しながら target 被験者への適応を進めたと考えられる。

## Tent に対する改善

EMA-Tent と通常 Tent の平均差分を以下に示す。

| Metric | EMA-Tent - Tent |
|---|---:|
| Accuracy | +0.2537 |
| Macro-F1 | +0.1937 |
| F1 class 0 | +0.3264 |
| F1 class 1 | +0.1445 |
| F1 class 2 | +0.1103 |

通常 Tent は current batch の BatchNorm 統計だけに依存するため、時系列順 batch の局所的なクラス偏りに強く影響された。EMA-Tent は過去 test batch の EMA 統計と source 統計を混合するため、current batch のみによる統計の揺れを抑制できる。この結果、通常 Tent に対して全指標で大きく改善した。

被験者別に見ると、EMA-Tent は Tent に対して 15被験者中14被験者で Macro-F1 を改善した。特に S5、S7、S15、S17、S16 で改善幅が大きかった。

| Subject | EMA-Tent - Tent Macro-F1 |
|---|---:|
| S5 | +0.4007 |
| S7 | +0.3869 |
| S15 | +0.3306 |
| S17 | +0.3171 |
| S16 | +0.2933 |

通常 Tent が大きく性能を落としていた被験者で、EMA-Tent がその破綻を大きく緩和している。

## OFTTA に対する改善

EMA-Tent と OFTTA の平均差分を以下に示す。

| Metric | EMA-Tent - OFTTA |
|---|---:|
| Accuracy | +0.1859 |
| Macro-F1 | +0.1408 |
| F1 class 0 | +0.2304 |
| F1 class 1 | +0.1156 |
| F1 class 2 | +0.0765 |

EMA-Tent は OFTTA に対しても平均的に優位であった。OFTTA は model weight を更新せず、BN統計補正とT3A型の分類器調整を行う。一方、EMA-Tent は BN affine パラメータを entropy minimization で更新しつつ、BN統計をEMAで安定化する。そのため、通常 Tent のような統計破綻を抑えながら、Tent の適応能力を一部保持できたと解釈できる。

被験者別では、EMA-Tent は OFTTA に対して 15被験者中14被験者で Macro-F1 を改善した。唯一、S2ではOFTTAがEMA-Tentを上回った。

## 被験者別の勝者

Macro-F1 に基づく各被験者の最良手法を以下に示す。

| Method | Best Subject Count |
|---|---:|
| Source | 3 |
| Tent | 0 |
| EMA-Tent | 11 |
| OFTTA | 1 |

EMA-Tent は 15被験者中11被験者で最良の Macro-F1 を示した。Source が最良だったのは S9、S11、S15 であり、OFTTA が最良だったのは S2 のみであった。

## 解釈

今回の結果は、WESAD の通常順序評価において、current batch 統計のみに依存する Tent が不安定であること、そして BN 統計に過去 test batch の EMA を導入することでその不安定性を大きく緩和できることを示している。

通常 Tent では、各 batch の BatchNorm 統計が現在 batch のみによって決まる。WESAD では評価データが実験状態ごとのブロック構造を持つため、current batch が特定クラスや特定状態に強く偏りやすい。この局所統計に基づいてBN正規化とentropy minimizationを行うと、モデルが一時的なブロックに過剰適応し、後続batchに対して性能を落とす可能性がある。

EMA-Tent では、BN統計を以下の3成分で構成する。

```text
source running stats
+ past test EMA stats
+ current batch stats
```

これにより、current batch の局所的な偏りを緩和しつつ、target 被験者から得た過去情報を蓄積できる。さらに、batch の予測分布が偏っている場合は current batch stats と entropy update を弱めるため、単一ブロックへの過剰適応を抑制できる。

## 注意点

EMA-Tent は平均では最良であったが、すべての被験者で Source を上回ったわけではない。S11、S15、S9では Source がEMA-Tentを上回った。これらの被験者では source model が既に十分良く、追加の適応が必ずしも有効ではなかった可能性がある。

また、S2では OFTTA が最良であった。S2では Tent と OFTTA がSourceから大きく改善しており、EMA-Tent の保守的な統計混合よりも、OFTTAの分類器調整が有効だったと考えられる。

## 論文向け要約

WESAD 3分類の通常順序評価において、通常 Tent は Macro-F1 0.3970 に留まり、Source の 0.5285 を下回った。これは、current batch の BatchNorm 統計のみに依存する Tent が、WESAD の時系列ブロック構造により局所的な batch 分布へ過剰適応したためと考えられる。

提案した EMA-Tent は、BatchNorm 統計を source running statistics、過去 test batch のEMA統計、現在 batch 統計の混合として扱うことで、current batch への過度な依存を抑制した。その結果、EMA-Tent は Accuracy 0.7090、Macro-F1 0.5908 を達成し、Source、Tent、OFTTA のすべてを平均性能で上回った。特に Stress F1 は Source の 0.4747 から 0.5785 へ、Amusement F1 は 0.2317 から 0.3213 へ改善した。

これらの結果は、WESADのようなブロック構造を持つ生体時系列データにおいて、テスト時BN統計を現在batchのみに依存させるのではなく、過去test batchの統計をEMAとして蓄積することが、オンラインTTAの安定化に有効であることを示している。
