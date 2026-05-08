# WESAD Batch Size 120: Class-Balanced Target Batch

## Purpose

This diagnostic experiment tests whether OFTTA recovers when every TTA batch contains all WESAD emotion labels.

The target loader was reordered using ground-truth labels so that each batch contains Baseline, Stress, and Amusement whenever possible. This is an oracle analysis condition for isolating batch-composition effects, not a deployable test-time setting.

## Settings

- Dataset: WESAD 3-class
- Checkpoint: `./ckpt_3class`
- TTA batch size: 120
- Methods: Source, Tent, OFTTA
- Target batch construction: class-balanced order using ground-truth labels

## Average Results

| Condition | Source Macro-F1 | Tent Macro-F1 | OFTTA Macro-F1 | Tent - Source | OFTTA - Source |
|---|---:|---:|---:|---:|---:|
| Sequential batch size 120 | 0.5285 | 0.3633 | 0.4022 | -0.1651 | -0.1262 |
| Shuffled batch size 120 | 0.5285 | 0.6334 | 0.6398 | +0.1049 | +0.1113 |
| Class-balanced batch size 120 | 0.5285 | 0.5783 | 0.5905 | +0.0498 | +0.0620 |

## Batch Composition Check

The class-balanced target loader removed the severe block-batch bias:

- `single_class_batch_ratio = 0.0` for all subjects.
- `mean_missing_classes = 0.0` for all subjects.
- Mean max-class ratio was around 0.51 to 0.52 for most subjects, instead of around 0.87 to 0.97 in the sequential condition.

Thus, every diagnostic batch contained all three labels.

## Interpretation

OFTTA failed in the sequential batch-size-120 condition but improved over Source when batches were forced to contain all classes. This supports the interpretation that WESAD's normal sequential block structure is a major cause of OFTTA failure.

However, class-balanced batching did not reach the shuffled condition. This suggests that merely including all classes in every batch is not the only factor. The exact class proportions, within-batch sample diversity, and random mixing across the whole target sequence also affect TTA behavior.

The result strengthens the paper's argument:

> OFTTA is not inherently incompatible with WESAD. Its failure in the normal WESAD order is caused mainly by the mismatch between OFTTA's target-batch assumption and WESAD's block-structured target sequence.

## Output Files

- Comparison result: `logs/wesad/compare_source_tent_oftta/260508_182200_WESAD3class_bs120_classBalancedBatch_SourceTentOFTTA/`
- Batch-bias check: `logs/wesad/batch_bias_analysis/260508_182630_WESAD3class_bs120_classBalancedBatch_biasCheck/`
