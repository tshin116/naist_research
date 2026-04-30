# EMA-Tent の Batch Composition 依存性分析

## 比較対象

通常順序評価:

```text
logs/wesad/compare_source_tent_oftta/260420_151858_EMATent手法導入後評価
```

shuffle評価:

```text
logs/wesad/compare_source_tent_oftta/260420_155012_EMATent手法導入後評価-shuffle
```

両評価は Source、Tent、EMA-Tent、OFTTA を比較している。通常順序評価では target 被験者の window を保存順に batch 化し、shuffle評価では target 被験者内の window を DataLoader で shuffle して batch 化している。

## Shuffle - Sequential の平均差分

各手法について、shuffle条件の平均性能から通常順序条件の平均性能を引いた。

| Method | ΔAccuracy | ΔMacro-F1 | ΔF1 class 0 | ΔF1 class 1 | ΔF1 class 2 |
|---|---:|---:|---:|---:|---:|
| Source | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Tent | +0.2747 | +0.2315 | +0.3262 | +0.2727 | +0.0955 |
| EMA-Tent | +0.0306 | +0.0005 | +0.0466 | -0.0024 | -0.0428 |
| OFTTA | +0.2343 | +0.1908 | +0.2575 | +0.2257 | +0.0893 |

通常 Tent は shuffle により Macro-F1 が +0.2315 改善し、OFTTA も +0.1908 改善した。一方、EMA-Tent の Macro-F1 差分は +0.0005 であり、ほぼ変化しなかった。

この結果から、EMA-Tent は通常 Tent や OFTTA と比較して batch composition 依存を大きく抑えていると解釈できる。通常 Tent と OFTTA は、batch がshuffleされて各batchのクラス構成が混ざることで大幅に改善したが、EMA-Tent は通常順序でもshuffle条件と同程度の性能を維持した。

## 被験者別の Shuffle Gain

Macro-F1 の `shuffle - sequential` を被験者ごとに集計した。

| Method | Mean Gain | Improved Subjects | Worsened Subjects | Min Gain | Max Gain |
|---|---:|---:|---:|---:|---:|
| Tent | +0.2315 | 15 | 0 | +0.1040 | +0.3993 |
| EMA-Tent | +0.0005 | 7 | 8 | -0.1398 | +0.1126 |
| OFTTA | +0.1908 | 15 | 0 | +0.0179 | +0.3997 |

Tent と OFTTA は全被験者で shuffle により改善した。これは、通常順序の batch 構成がこれらの手法に不利であることを示している。一方、EMA-Tent は改善被験者が7人、悪化被験者が8人であり、平均差分はほぼゼロであった。これは、EMA-Tent が通常順序の段階で batch 構成の偏りを補正しており、shuffleによる追加改善が小さいことを示す。

## Tent vs EMA-Tent

通常順序条件における EMA-Tent と Tent の Macro-F1 差分は以下である。

| Comparison | Mean Difference | Improved Subjects | Worsened Subjects |
|---|---:|---:|---:|
| EMA-Tent - Tent | +0.1937 | 14 | 1 |

通常順序では、EMA-Tent は15被験者中14被験者で Tent を上回った。平均 Macro-F1 差分は +0.1937 であり、通常 Tent の破綻を大きく緩和している。

改善幅が大きい被験者は以下である。

| Subject | EMA-Tent - Tent Macro-F1 |
|---|---:|
| S5 | +0.4007 |
| S7 | +0.3869 |
| S15 | +0.3306 |
| S17 | +0.3171 |
| S16 | +0.2933 |

これらは通常 Tent が大きく性能を落としていた被験者であり、EMA-Tent の BN統計安定化が有効に働いたと考えられる。

shuffle条件では、EMA-Tent は Tent に対して平均 -0.0373 であり、15被験者中6被験者で改善、9被験者で悪化した。shuffle条件では各batchがtarget分布をより代表しやすくなるため、current batch statistics に強く依存する通常 Tent も安定しやすい。そのため、EMA-Tent の優位性は主に通常順序、すなわちオンライン時系列評価に近い条件で現れる。

## 結論

EMA-Tent は batch composition 依存を抑えていると言える。根拠は以下である。

1. Tent の `shuffle - sequential Macro-F1` は +0.2315、OFTTA は +0.1908 であった。
2. EMA-Tent の `shuffle - sequential Macro-F1` は +0.0005 であり、ほぼゼロであった。
3. 通常順序条件で、EMA-Tent は Tent に対して 15被験者中14被験者で改善した。
4. 通常順序条件での EMA-Tent - Tent の平均 Macro-F1 差分は +0.1937 であった。

したがって、EMA-Tent は current batch statistics のみに依存する通常 Tent の破綻を緩和し、WESAD のブロック構造に対してより安定したオンラインTTAとして機能していると考えられる。
