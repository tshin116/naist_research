# WESAD 2-class Stress / No-stress evaluation: EMA-Tent analysis

## 実験条件

- Dataset: WESAD
- Task: 2-class classification
  - No-stress: WESAD raw labels 1, 3, 4
  - Stress: WESAD raw label 2
- Model: fixed LOSO 1D-CNN checkpoints in `ckpt_fixed`
- Metric: Macro-F1 over the two classes
- Batch size: 64
- Compared methods: Source, Norm, Tent, OFTTA, EMA-Tent

## 実行ログ

- Sequential order:
  - `logs/wesad/compare_source_tent_oftta/260616_075205_WESAD2class_binaryCache_SourceNormTentOFTTAEMATent`
- Shuffled target order:
  - `logs/wesad/compare_source_tent_oftta/260616_075208_WESAD2class_binaryCache_shuffle_SourceNormTentOFTTAEMATent`

2クラス用のprocessed cacheは、古いcacheと混ざらないように `data/wesad/processed_binary` に分けた。

## Average performance

### Sequential order

| Method | Macro-F1 | No-stress F1 | Stress F1 |
|---|---:|---:|---:|
| Source | 0.6737 ± 0.1968 | 0.8740 | 0.4734 |
| Norm | 0.6063 ± 0.0627 | 0.8433 | 0.3694 |
| Tent | 0.6061 ± 0.0627 | 0.8433 | 0.3690 |
| OFTTA | 0.5646 ± 0.0987 | 0.7096 | 0.4195 |
| EMA-Tent | **0.7672 ± 0.1656** | **0.8904** | **0.6441** |

### Shuffled target order

| Method | Macro-F1 | No-stress F1 | Stress F1 |
|---|---:|---:|---:|
| Source | 0.6737 ± 0.1968 | 0.8740 | 0.4734 |
| Norm | 0.7258 ± 0.1851 | 0.8707 | 0.5810 |
| Tent | 0.7258 ± 0.1858 | 0.8708 | 0.5808 |
| OFTTA | 0.7261 ± 0.1558 | 0.8367 | 0.6154 |
| EMA-Tent | **0.7429 ± 0.1780** | **0.8849** | 0.6010 |

## Sequential orderでの差分

| Comparison | Mean delta Macro-F1 | Wins / Losses |
|---|---:|---:|
| EMA-Tent - Source | +0.0935 | 9 / 6 |
| EMA-Tent - Norm | +0.1609 | 13 / 2 |
| EMA-Tent - Tent | +0.1611 | 13 / 2 |
| EMA-Tent - OFTTA | +0.2027 | 14 / 1 |

Wilcoxon signed-rank test, two-sided:

| Comparison | p-value |
|---|---:|
| EMA-Tent vs Source | 0.083252 |
| EMA-Tent vs Norm | 0.001526 |
| EMA-Tent vs Tent | 0.001526 |
| EMA-Tent vs OFTTA | 0.000183 |

EMA-TentはTent, Norm, OFTTAに対して有意に高い。一方でSourceとの差は平均では+0.0935あるが、p=0.083であり、5%水準では有意とは言い切れない。これはS5, S17, S13, S8, S9のようにSourceがすでに強い被験者が存在するためである。

## Shuffle effect

`Shuffle Macro-F1 - Sequential Macro-F1`:

| Method | Mean delta | Positive / Negative |
|---|---:|---:|
| Source | +0.0000 | 0 / 0 |
| Norm | +0.1195 | 12 / 3 |
| Tent | +0.1197 | 12 / 3 |
| OFTTA | +0.1615 | 15 / 0 |
| EMA-Tent | -0.0243 | 6 / 9 |

Norm, Tent, OFTTAはshuffleで大きく改善した。これは通常順序のバッチ構成がTTAの性能を強く悪化させていることを示す。一方でEMA-Tentは通常順序でも高く、shuffleによる平均改善は出ていない。つまりEMA-Tentは、通常順序に含まれるブロック構造の悪影響をかなり吸収している。

## Interpretation

2クラス化すると、Stress / No-stressの境界が3クラスより単純になるため、Sourceだけでも平均Macro-F1は0.6737まで上がる。しかし、通常順序でNorm/Tent/OFTTAを適用すると平均Macro-F1はSourceより低下する。特にTentはStress F1が0.3690まで落ちており、通常順序の偏ったバッチ統計がStress検出を崩している。

EMA-TentはSequential orderでStress F1を0.6441まで改善した。これは、現バッチ統計だけに依存せず、source統計と過去target EMA統計を混合することで、単一クラスに近いバッチが連続する状況でもBN統計が急激に片方へ引きずられにくくなったためと解釈できる。

ただし、Sourceが非常に強い被験者ではEMA-Tentが必ずしも上回らない。S5ではSource 0.9902に対してEMA-Tent 0.9753、S17ではSource 0.9122に対してEMA-Tent 0.8456であった。したがって、2クラス条件では「EMA-Tentは既存TTAの破綻を大きく抑えるが、全被験者でSourceを超えるとは限らない」と書くのが妥当である。

## Paper-ready statement

WESADをStress / No-stressの2クラス分類として評価した場合、通常順序条件においてSourceは平均Macro-F1 0.6737を示した。一方、NormおよびTentはそれぞれ0.6063, 0.6061まで低下し、OFTTAも0.5646にとどまった。これは、2クラス化しても通常順序のブロック構造に起因するバッチ統計の偏りが既存TTAを不安定化することを示している。これに対し、EMA-Tentは平均Macro-F1 0.7672を達成し、Tentに対して+0.1611、OFTTAに対して+0.2027の改善を示した。Wilcoxon signed-rank testでもEMA-TentはTentおよびOFTTAより有意に高かった（Tent: p=0.001526, OFTTA: p=0.000183）。特にStress F1はTentの0.3690からEMA-Tentの0.6441へ改善しており、EMA-Tentは偏った逐次バッチにおけるStress検出の崩壊を抑制したと考えられる。

一方、EMA-TentとSourceの差は平均では+0.0935であったが、p=0.083252であり5%水準では有意ではなかった。したがって、2クラス条件では、EMA-Tentは既存TTA手法に対して明確な改善を示すが、Sourceを全被験者で一貫して上回る手法というより、通常順序条件でTTAが破綻する被験者に対して有効な安定化手法として位置付けるのが適切である。
