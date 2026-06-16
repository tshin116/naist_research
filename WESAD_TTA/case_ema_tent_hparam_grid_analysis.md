# CASE EMA-Tent Hyperparameter Grid Analysis

## 実験目的

WESADで用いたEMA-TentのBN統計混合比をそのままCASEへ流用するのではなく、CASE Valence/Arousalに対して適した混合比を探索した。

比較した主な軸は以下である。

- `ema_tent_source_weight`
- `ema_tent_ema_weight`
- `ema_tent_batch_weight`
- `ema_tent_momentum`

なお、EMA-Tent実装では混合比は内部で正規化される。またgateが小さい場合、`batch_weight * (1 - g)` はsource側へ戻されるため、`source_weight=0.0` の設定でも、gateが働く局面では完全にsource統計が消えるわけではない。

## 実行ログ

Source学習:

- Valence: `logs/case/train_fixed_loso/260616_014148/loso_fixed_epoch_results.csv`
- Arousal: `logs/case/train_fixed_loso/260616_014149/loso_fixed_epoch_results.csv`

基本比較:

- Valence: `logs/case/compare_source_tent_oftta/260616_053356_CASE_valence_SourceNormTentOFTTAEMATent/`
- Arousal: `logs/case/compare_source_tent_oftta/260616_053358_CASE_arousal_SourceNormTentOFTTAEMATent/`

EMA-Tent coarse grid:

- Valence: `logs/case/compare_source_tent_oftta/260616_060943_CASE_valence_EMATentHparamGrid/`
- Arousal: `logs/case/compare_source_tent_oftta/260616_061001_CASE_arousal_EMATentHparamGrid/`

EMA-Tent focused grid:

- Valence: `logs/case/compare_source_tent_oftta/260616_061218_CASE_valence_EMATentFocusedGrid/`
- Arousal: `logs/case/compare_source_tent_oftta/260616_061238_CASE_arousal_EMATentFocusedGrid/`

## 結果概要

Macro-F1平均の上位は以下であった。

| Method / setting | Valence Macro-F1 | Arousal Macro-F1 | 2-task mean |
| --- | ---: | ---: | ---: |
| Tent | 0.5670 | 0.5622 | 0.5646 |
| Norm | 0.5671 | 0.5618 | 0.5644 |
| OFTTA | 0.5544 | 0.5457 | 0.5500 |
| EMA-Tent `source=0.0, ema=0.5, batch=0.5, momentum=0.70` | 0.5457 | 0.5342 | 0.5400 |
| EMA-Tent `source=0.1, ema=0.5, batch=0.4, momentum=0.70` | 0.5437 | 0.5255 | 0.5346 |
| EMA-Tent `source=0.2, ema=0.5, batch=0.3, momentum=0.60` | 0.5417 | 0.5276 | 0.5346 |
| Source | 0.4545 | 0.4073 | 0.4309 |

## EMA-Tent内の最良設定

今回試した範囲では、CASEに対するEMA-Tentの最良設定は以下であった。

```text
ema_tent_source_weight: 0.0
ema_tent_ema_weight: 0.5
ema_tent_batch_weight: 0.5
ema_tent_momentum: 0.70
```

この設定の結果:

```text
Valence Macro-F1: 0.5457
Arousal Macro-F1: 0.5342
2-task mean: 0.5400
```

被験者別の勝敗:

| Task | vs Source | vs Norm | vs Tent | vs OFTTA |
| --- | ---: | ---: | ---: | ---: |
| Valence | 21/30 | 11/30 | 11/30 | 15/30 |
| Arousal | 24/30 | 12/30 | 12/30 | 16/30 |

## 解釈

CASEでは、WESADで有効だった保守的なsource anchoringは最適ではなかった。`source`基礎重みを大きくした設定、例えば `source=0.5, ema=0.4, batch=0.1, momentum=0.90` は、Valence/Arousalの両方で低かった。

一方で、`source`を弱め、`current batch`と`EMA`を強めた設定がEMA-Tent内では最も良かった。これはCASEでは現在target batchの統計が比較的有用であり、WESADのようにsource統計を強く残す必要が小さいことを示唆する。

ただし、全手法で見ると、最良はNorm/Tentであり、EMA-Tent最良設定はそれに届いていない。したがってCASEについては、現時点では「EMA-Tentが最良」とは言えない。むしろ、CASEでは通常BN適応、すなわちcurrent batch統計を強く使う方向が有効である可能性が高い。

## 結論

CASE用EMA-Tent設定として採用候補にするなら、現時点では以下が最も妥当である。

```text
source=0.0, ema=0.5, batch=0.5, momentum=0.70
```

ただし論文上は、CASEではTent/NormがEMA-Tentより高く、EMA-TentはWESADのブロック構造問題に対する手法としての性質が強い、という整理が自然である。
