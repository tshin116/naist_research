# EMA-Tent Gate 診断分析

## 目的

このブランチでは、現在の EMA-Tent の gate が WESAD に対して保守的すぎるかを確認した。

懸念点は、WESAD の target batch が実験プロトコル由来のブロック構造により自然にクラス偏りを持つことである。もし gate が「クラス偏りのある batch = 信頼できない batch」と解釈すると、current batch の BN 統計量が必要以上に抑制される可能性がある。

## 使用ログ

- 通常順序: `logs/wesad/compare_source_tent_oftta/260525_034958_WESAD3class_EMATent_gate_diagnostics_sequential`
- Shuffle 順序: `logs/wesad/compare_source_tent_oftta/260525_035107_WESAD3class_EMATent_gate_diagnostics_shuffle`

各ログには以下が含まれる。

- `source_tent_oftta_comparison.csv`
- `ema_tent_batch_diagnostics.csv`
- `ema_tent_gate_summary.md`

## 主結果

### 通常順序

| Method | Average Macro-F1 |
| --- | ---: |
| Source | 0.5285 |
| Tent | 0.3970 |
| EMATent | 0.6136 |
| EMATentProbe025 | 0.6112 |
| EMATentProbe050 | 0.6099 |
| EMATentProbe100 | 0.6111 |
| EMATentMinGate025 | 0.6126 |
| EMATentMinGate050 | 0.6131 |

デフォルト EMA-Tent の `raw_gate` 平均は 0.4374 であった。基本設定では current batch weight は 0.3 であるため、実効的な current batch BN weight は平均 0.1312 まで低下した。中央値は 0.1208、25 パーセンタイルは 0.0334 であった。

したがって、通常順序の WESAD では、現在の gate が current batch 統計量をかなり強く抑制していることが確認できる。

### Shuffle 順序

| Method | Average Macro-F1 |
| --- | ---: |
| Source | 0.5285 |
| Tent | 0.6285 |
| EMATent | 0.6111 |
| EMATentProbe025 | 0.6123 |
| EMATentProbe050 | 0.6134 |
| EMATentProbe100 | 0.6144 |
| EMATentMinGate025 | 0.6123 |
| EMATentMinGate050 | 0.6135 |

Shuffle 順序では、デフォルト EMA-Tent の `raw_gate` 平均は 0.8238 であった。実効的な current batch BN weight は平均 0.2471 であり、設定上の最大値 0.3 に比較的近い。中央値は 0.2592 であった。

これは、batch が target 分布を代表している場合には current batch BN 統計量が有効であることを示している。また shuffle 条件では Tent が EMA-Tent より高いため、current batch が信頼できる状況では EMA-Tent の保守的な gate が適応を制限する可能性がある。

## Gate の挙動

デフォルト EMA-Tent の gate と真の batch class bias には以下の相関があった。

| Condition | corr(true max class ratio, raw gate) | corr(num present classes, raw gate) |
| --- | ---: | ---: |
| Sequential | -0.4376 | 0.4330 |
| Shuffle | -0.2200 | -0.0015 |

通常順序では、真のラベル構成が偏るほど gate が小さくなる。この挙動自体は設計意図と一致している。しかし WESAD では、ラベル偏りは偶然のノイズではなく、感情提示プロトコルのブロック構造そのものである。そのため、長い感情ブロックでは gate が過度に保守的になる可能性がある。

## 解釈

現在の EMA-Tent は、通常順序 WESAD で見られる Tent の破綻を緩和できている。ただし診断結果を見ると、その改善は current batch BN 統計量への依存を大きく下げることで得られている側面がある。

つまり EMA-Tent は、target subject の各 block から積極的に適応しているというより、source 統計量と過去 target EMA 統計量に寄せることで安定化している可能性がある。

強い根拠は以下である。

- 通常順序では Tent が低下する: Macro-F1 0.3970。
- 通常順序では EMA-Tent が改善する: Macro-F1 0.6136。
- Shuffle 順序では Tent が強い: Macro-F1 0.6285。
- Shuffle 順序では EMA-Tent は Tent より低い: Macro-F1 0.6111。
- 通常順序のデフォルト EMA-Tent では実効 current batch weight が低い: mean 0.1312, median 0.1208。
- Shuffle 順序のデフォルト EMA-Tent では実効 current batch weight が高い: mean 0.2471, median 0.2592。

したがって、「current batch 統計量が悪い」と結論づけるのは不正確である。より正確には以下である。

> Current batch 統計量は、batch が target 分布を代表している場合には有効である。これは shuffle 条件で Tent が高性能になることから確認できる。一方で通常順序の WESAD では、連続する class-biased block により current-batch-only normalization が不安定になる。EMA-Tent はこの破綻を緩和するが、現在の gate は自然な感情 block に対しても current batch 統計量を強く抑制するため、過度に保守的である可能性がある。

## まだ切り分けが必要な点

追加で `Source / Norm / Tent` 比較を行い、通常順序における Tent の低下要因を分解した。

追加ログは以下である。

- 通常順序: `logs/wesad/compare_source_tent_oftta/260526_003406_WESAD3class_SourceNormTent_sequential_BNstats_vs_affineUpdate`
- Shuffle 順序: `logs/wesad/compare_source_tent_oftta/260526_003406_WESAD3class_SourceNormTent_shuffle_BNstats_vs_affineUpdate`

Tent には少なくとも次の 2 要素がある。

1. BN 層で current batch 統計量を 100% 使う。
2. Entropy minimization により BN affine parameter、すなわち `gamma` と `beta` を更新する。

したがって、Tent の低下が以下のどれに由来するかは追加検証が必要である。

- current batch BN 統計量だけが悪い。
- `gamma` / `beta` 更新だけが悪い。
- current batch BN 統計量で偏った特徴分布を作り、その上で `gamma` / `beta` 更新が累積するため両方が悪い。

この切り分けには、`Source / Norm / Tent` 比較が必要である。

- `Source`: source running BN 統計量を使う。test-time update はしない。
- `Norm`: current batch BN 統計量を使うが、`gamma` / `beta` は更新しない。
- `Tent`: current batch BN 統計量を使い、さらに `gamma` / `beta` を entropy minimization で更新する。

判定は以下のように行う。

- `Norm` も `Tent` も低い場合: current batch BN 統計量の偏りが主因である可能性が高い。
- `Norm` は高いが `Tent` だけ低い場合: `gamma` / `beta` 更新の累積が主因である可能性が高い。
- `Norm` が Source より少し低く、`Tent` がさらに低い場合: BN 統計量の偏りと `gamma` / `beta` 更新の両方が悪化に寄与している可能性が高い。

### Source / Norm / Tent 追加実験

| Condition | Source Macro-F1 | Norm Macro-F1 | Tent Macro-F1 | Norm - Source | Tent - Norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sequential | 0.5285 | 0.3972 | 0.3970 | -0.1312 | -0.0002 |
| Shuffle | 0.5285 | 0.6284 | 0.6285 | +0.1000 | +0.0001 |

この結果は非常に重要である。通常順序では、`Norm` の時点で `Source` より大きく低下している。一方で `Tent` は `Norm` とほぼ同じであり、`Tent - Norm` は平均 -0.0002 に過ぎない。

したがって、今回の WESAD 通常順序における Tent の低下主因は、`gamma` / `beta` 更新ではなく、current batch BN 統計量を 100% 使用することにあると考えられる。

また shuffle 条件では、`Norm` が `Source` より大きく改善している。これは、current batch BN 統計量そのものが常に悪いわけではなく、batch が target 分布を代表している場合にはむしろ有効であることを示している。

以上より、現時点での結論は以下である。

> WESAD 通常順序で Tent が低下する主因は、`gamma` / `beta` 更新ではなく、ブロック構造により偏った current batch の BN 統計量を 100% 使用することである。Shuffle 条件では current batch が複数クラスを代表するため、同じ current-batch normalization が性能改善に寄与する。

ただし、今回の `Tent` は 1 step 設定であり、各 batch の出力はその batch の forward 後、次 batch に向けて `gamma` / `beta` を更新する流れである。今回の実験では `Tent` と `Norm` がほぼ一致したため、少なくとも現在の設定では `gamma` / `beta` 更新の追加的な影響は非常に小さい。

## 次の設計方針

現在のデフォルト EMA-Tent は、安定化ベースラインとしては妥当である。しかし最終設計として最適とは限らない。理想的には以下の両極端を避ける必要がある。

- Tent: current batch 100%。通常順序の block 構造では攻めすぎる。
- 現在の EMA-Tent: block batch で current batch contribution が低くなりすぎる可能性がある。

次の候補は、current batch 統計量を最低限残す設計、または「有効な単一感情 block」と「危険な予測崩壊」を区別できる block-aware / transition-aware gate である。単純な `min_gate` ablation は平均性能を明確には改善しなかったため、単に下限を上げるよりも、block 構造を明示的に扱う gate の方が有望である。
