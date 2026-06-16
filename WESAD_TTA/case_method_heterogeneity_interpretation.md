# CASEにおける手法ごとの被験者差の考察

## 要点

CASEでは、被験者によって有効なTTA手法がかなり異なる。特にArousalでは、S1, S4, S6, S11, S20, S30のようにEMA-Tentが有効な被験者がある一方で、S2, S3, S8, S17, S18, S23, S28のようにNorm/Tent/OFTTAが強い被験者も存在した。

この差は、単純なbatch内ラベル偏りだけでは説明しきれない。より強い説明は、「current batch統計をそのまま使うことが有効な被験者か、それとも危険な被験者か」という違いである。

## 観測された関係

Arousalでは、default EMA-Tentの改善量と、既存TTAの改善量に強い負の相関があった。

```text
target: EMATent_gain_vs_best_baseline

OFTTA_gain_vs_source:
  Spearman r = -0.631, p = 0.000187

Norm_gain_vs_source:
  Spearman r = -0.556, p = 0.001424

Tent_gain_vs_source:
  Spearman r = -0.550, p = 0.001634
```

つまり、Norm/Tent/OFTTAがsourceから大きく改善できる被験者では、EMA-Tentは相対的に負けやすい。逆に、Norm/Tent/OFTTAがsourceからあまり改善しない、あるいは悪化する被験者では、EMA-Tentが有効になりやすい。

これはEMA-Tentの設計と整合的である。EMA-Tentはsource統計、過去target統計、current batch統計を混合し、さらにgateでcurrent batch寄与を抑制する。そのため、current batch統計を強く使うべき被験者では保守的すぎる。一方で、current batch統計をそのまま使うと崩れる被験者では安定化が効く。

## EMA-Tentが有効な被験者

### S1 Arousal

S1ではSourceがMacro-F1 0.340で、Norm/Tentは0.317まで下がった。OFTTAも0.332でsourceを超えられていない。一方、default EMA-Tentは0.351、CASE最適EMA-Tentは0.428まで改善した。

この被験者では、current batch統計をそのまま使うNorm/Tentが有効ではない。つまり、target batch統計が分類境界を良い方向に動かすというより、むしろ不安定化している可能性が高い。EMA-Tentはcurrent batch統計への依存を抑えつつ、過去target統計を蓄積するため、過剰適応を避けられたと考えられる。

### S4 Arousal

S4ではSource 0.396、Norm/Tent 0.509、OFTTA 0.414に対して、default EMA-Tentは0.577、CASE最適EMA-Tentは0.623で最も高かった。

S4はArousalで単一クラスbatchが2個あり、batch構成の偏りが強い。Norm/Tentはcurrent batch統計に直接寄るため、局所的なbatch構成に引っ張られやすい。EMA-Tentはcurrent batch統計を混合成分の一部に制限するため、偏ったbatch区間での統計崩壊を抑えた可能性がある。

### S6 Arousal

S6ではSource 0.399、Norm/Tent 0.450、OFTTA 0.415に対して、default EMA-Tentは0.521、CASE最適EMA-Tentは0.566であった。

S6は平均majority ratioが0.777で、単一クラスbatchも1個ある。batch偏りは強いが、default EMA-Tentのgate平均は0.966と高く、実効batch重みも0.290と高い。これは、予測分布としてはbatch統計を使えると判断されていたことを意味する。単純にbatchを抑えるのではなく、過去target EMAと併用して平滑化したことが効いた可能性がある。

### S11 Arousal

S11ではSource 0.563、Norm/Tent 0.553、OFTTA 0.567に対して、default EMA-Tentは0.594、CASE最適EMA-Tentは0.600であった。

S11は単一クラスbatchがなく、平均majority ratioも0.589で比較的バランスが良い。この場合、EMA-Tentは「偏ったbatchを避ける」というより、target統計を逐次的に平滑化して蓄積する効果が効いていると考えられる。current batchだけに依存するNorm/Tentより、過去target情報を保持する方が安定した可能性がある。

### S30 Arousal

S30ではSource 0.386、Norm/Tent 0.562、OFTTA 0.598に対して、default EMA-Tentは0.691で最良であった。ただしCASE最適EMA-Tentは0.614で、defaultより低い。

この被験者は、current batch統計もある程度有効だが、OFTTAよりさらに過去target統計を使うdefault EMA-Tentが良かった例である。S30ではsourceからの改善余地が大きく、かつcurrent batchのみに寄せるよりEMAで蓄積したtarget統計の方が有効だったと考えられる。

## Norm/Tent/OFTTAが強い被験者

### S2 Arousal

S2ではSource 0.209に対して、Norm/Tentが0.659、OFTTAが0.620まで大きく改善した。一方、default EMA-Tentは0.209でsourceと同じ、CASE最適EMA-Tentも0.195で改善しなかった。

これは非常に重要な例である。S2ではcurrent batch統計を使うことが強く有効だった。しかしdefault EMA-Tentのgate平均は0.169、実効batch重みは0.051と非常に低く、current batch統計をほとんど使っていない。そのため、sourceに近い統計に拘束され、Norm/Tentで得られる大きな改善を逃したと考えられる。

つまりS2では、EMA-Tentの保守性が裏目に出ている。

### S3 Arousal

S3ではSource 0.425に対して、Norm 0.685、Tent 0.681、OFTTA 0.595であり、current batch統計を使うNorm/Tentが非常に強い。一方、default EMA-Tentは0.558で、改善はするがNorm/Tentには届かない。

この被験者では、target batch統計がsourceとの差を補正する有用な情報になっている。EMA-Tentは統計混合によりこの補正を弱めるため、Norm/Tentに負けたと考えられる。

### S8 Arousal

S8ではSource 0.501、Norm/Tent 0.663、OFTTA 0.721に対して、default EMA-Tentは0.539であった。

OFTTAが最も強いことから、この被験者ではsource情報を完全に捨てる必要はないが、current batch側の情報も強く使う必要があると考えられる。EMA-Tentは過去EMAとgateにより変化を滑らかにするため、OFTTAが捉えたようなtarget側への十分な移動が弱くなった可能性がある。

### S23 Arousal

S23ではSource 0.265、Norm/Tent 0.723、OFTTA 0.724に対し、default EMA-Tentは0.436であった。CASE最適EMA-Tentでは0.664まで改善したが、それでもNorm/Tent/OFTTAには届いていない。

この被験者は、sourceが大きく外れており、target batch統計を積極的に使う必要がある例である。EMA-Tentの保守性を弱めることで改善するが、最良は依然としてNorm/Tent/OFTTAである。

## batch内ラベル偏りだけでは説明できない理由

CASEでは、batch内ラベル偏りは確かに存在する。

```text
Valence:
  single-class batches = 13 / 137
  mean majority ratio = 0.720

Arousal:
  single-class batches = 18 / 137
  mean majority ratio = 0.704
```

しかし、ArousalでEMA-Tentの改善量とbatch偏り指標の相関は強くなかった。例えばdefault EMA-Tentの改善量に対して、single_class_batchesやmean majority ratioは主要な説明変数ではなかった。

一方で、Norm/Tent/OFTTAのsourceからの改善量は強い負の相関を示した。これは、「batchが偏っているか」よりも、「その被験者ではcurrent batch統計を使うことが有効かどうか」の方が支配的であることを意味する。

同じ単一クラスbatchがあっても、S4/S6のようにEMA-Tentが効く場合と、S2/S8のようにNorm/Tent/OFTTAが効く場合がある。そのため、単一クラスbatchの存在だけで手法の優劣を説明するのは不十分である。

## ValenceとArousalの違い

Valenceでは、default EMA-Tentが明確に勝つ被験者はArousalより少ない。S4, S17, S25, S29, S30などでは有効だが、全体平均ではNorm/Tentの方が高い。

Valenceでは、full single-class batchの有無がdefault EMA-Tentの改善とやや正の関係を持っていた。ただしサンプル数は少なく、統計的には強い結論ではない。

Arousalでは、current batch統計を使うNorm/Tentが強い被験者と、EMAによる平滑化が効く被験者の差がより明確であった。

## 総合解釈

CASEにおける手法差は、以下の3タイプに分けられる。

### 1. current batch統計が強く有効な被験者

例: S2, S3, S8, S23, S28

このタイプではNorm/Tent/OFTTAが強い。source統計からtarget batch統計へ大きく動かすことが必要であり、EMA-Tentのsource/EMA混合やgateによる抑制は保守的すぎる。

### 2. current batch統計が危険で、平滑化が有効な被験者

例: S1, S4, S6, S20, S30

このタイプではEMA-Tentが有効である。Norm/Tentのようにcurrent batch統計へ直接寄せると不安定になり、source統計または過去target EMAを混ぜることで安定化する。

### 3. sourceがすでに強い、またはTTA自体が不要な被験者

例: 一部のValence被験者

このタイプでは、どのTTAもsourceを大きく上回らない。EMA-Tentも過度な変化を避ける点では有利だが、平均性能では大きな差になりにくい。

## 論文向けの示唆

CASEの結果は、EMA-Tentが常に最良という主張にはならない。一方で、EMA-Tentが有効な被験者群が存在することは明確である。

したがって、論文では以下のように整理するのが妥当である。

```text
CASEでは、current batch統計を直接用いるNorm/Tentが平均的には最も高い性能を示した。
一方で、被験者によってはcurrent batch統計への急激な適応が不安定化を招き、
EMA-Tentのように過去target統計を蓄積しながら統計変化を平滑化する手法が有効であった。
このことは、感情推定TTAではデータセット単位だけでなく被験者単位で最適な適応強度が異なることを示している。
```

より発展的には、全被験者で単一のTTA手法を固定するのではなく、source予測の不安定性、batch予測分布、gate値、target batch統計の変動量を用いて、Norm/Tent型の強い適応とEMA-Tent型の保守的適応を切り替えるmeta-adaptationが有望である。
