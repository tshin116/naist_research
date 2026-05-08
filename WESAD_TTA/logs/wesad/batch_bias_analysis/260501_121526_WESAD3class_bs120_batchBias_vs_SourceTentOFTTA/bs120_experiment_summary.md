# Batch Size 120 Additional Experiments

## Purpose

The goal was to separate three possible factors behind the poor OFTTA performance on WESAD in the normal target order:

- target-batch class composition bias caused by the WESAD block protocol,
- TTA batch-size dependence,
- differences from HAR datasets where OFTTA works well.

All WESAD experiments used the same trained 3-class checkpoints and changed only the TTA target-loader condition.

## WESAD: Batch Size 120

| Condition | Source Macro-F1 | Tent Macro-F1 | OFTTA Macro-F1 | OFTTA - Source |
|---|---:|---:|---:|---:|
| Sequential order | 0.5285 | 0.3633 | 0.4022 | -0.1262 |
| Shuffled order | 0.5285 | 0.6334 | 0.6398 | +0.1113 |

With the same batch size of 120, OFTTA fails in the sequential condition but improves clearly in the shuffled condition. This indicates that the poor sequential-order result cannot be explained by batch size alone. The target batch composition induced by the WESAD block structure is a major factor.

## WESAD Batch Bias Correlation

For batch size 120, each subject had 4 target batches. The batch-bias summary showed:

- `single_class_batch_ratio = 0.5` for all subjects.
- `mean_missing_classes = 1.5` for all subjects.
- Mean max-class ratio was very high for most subjects, typically around 0.87 to 0.97.

Because `single_class_batch_ratio` and `mean_missing_classes` were constant across subjects, their correlations are undefined. This itself is informative: the sequential WESAD target loader is consistently biased across subjects at batch size 120.

The OFTTA shuffle gain had a negative Pearson correlation with mean max-class ratio:

| Target | Bias metric | Pearson r | Spearman rho |
|---|---|---:|---:|
| OFTTA shuffle gain | mean max class ratio | -0.3588 | -0.2163 |
| OFTTA shuffle gain | mean imbalance | -0.2557 | -0.1857 |

The main evidence is therefore the large sequential-vs-shuffle gap, rather than a strong subject-level monotonic correlation.

## HAR: OFTTA With Default Batch Size vs Batch Size 120

| Dataset | Setting | Source Macro-F1 | OFTTA Macro-F1 | OFTTA - Source |
|---|---|---:|---:|---:|
| UCI | default batch | 0.8546 | 0.9145 | +0.0599 |
| UCI | batch 120 | 0.8546 | 0.8666 | +0.0120 |
| OPPO | default batch | 0.6975 | 0.7263 | +0.0288 |
| OPPO | batch 120 | 0.6975 | 0.7162 | +0.0187 |
| UNIMIB | default batch | 0.4753 | 0.5356 | +0.0604 |
| UNIMIB | batch 120 | 0.4753 | 0.5098 | +0.0346 |

Reducing the HAR TTA batch size to 120 weakens OFTTA's improvement for all three HAR datasets. However, OFTTA still improves the average Macro-F1 over Source in UCI, OPPO, and UNIMIB. This suggests that batch size contributes to OFTTA stability, but it is not sufficient to reproduce the severe WESAD sequential-order failure.

## Interpretation

These results support the following interpretation.

OFTTA can work when the target batch is a reasonable local estimate of the target domain distribution. This is approximately true in HAR, especially when batches contain many windows and multiple classes. When HAR batch size is reduced to 120, OFTTA becomes weaker but does not collapse on average.

In WESAD, the sequential target order creates emotion-block batches. Even with batch size 120, half of the batches are single-class batches for every subject, and batches miss 1.5 out of 3 classes on average. Under this condition, the target batch no longer represents the target subject distribution. Therefore, OFTTA's target-batch adaptation is driven by block-local class composition rather than stable subject-specific distribution shift.

The shuffled WESAD result is the key control: using the same model and same batch size, simply changing the target order turns OFTTA from worse than Source to better than Source. This indicates that the main failure mode is the interaction between OFTTA and WESAD's sequential block structure, with batch size acting as a secondary factor.

## Output Files

- Sequential WESAD comparison: `logs/wesad/compare_source_tent_oftta/260501_120802_WESAD3class_bs120_sequential_SourceTentOFTTA/`
- Shuffled WESAD comparison: `logs/wesad/compare_source_tent_oftta/260501_121327_WESAD3class_bs120_shuffle_SourceTentOFTTA/`
- Batch-bias analysis: `logs/wesad/batch_bias_analysis/260501_121526_WESAD3class_bs120_batchBias_vs_SourceTentOFTTA/`
- HAR OFTTA default batch metrics: `/work/shinsaku-t/naist_reserch/OFTTA/logs/har_oftta_metrics/oftta_default_batch_metrics.csv`
- HAR OFTTA batch 120 metrics: `/work/shinsaku-t/naist_reserch/OFTTA/logs/har_oftta_metrics/oftta_bs120_metrics.csv`
