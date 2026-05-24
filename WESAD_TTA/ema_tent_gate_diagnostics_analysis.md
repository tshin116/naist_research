# EMA-Tent Gate Diagnostics

## Purpose

This branch checks whether the current EMA-Tent gate is too conservative for WESAD.
The concern is that WESAD target batches are naturally class-biased because of the block-structured protocol. If the gate interprets every class-biased batch as unreliable, the current-batch BN statistics may be suppressed too strongly.

## Logs

- Sequential order: `logs/wesad/compare_source_tent_oftta/260525_034958_WESAD3class_EMATent_gate_diagnostics_sequential`
- Shuffle order: `logs/wesad/compare_source_tent_oftta/260525_035107_WESAD3class_EMATent_gate_diagnostics_shuffle`

Each log contains:

- `source_tent_oftta_comparison.csv`
- `ema_tent_batch_diagnostics.csv`
- `ema_tent_gate_summary.md`

## Main Results

### Sequential Order

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

For the default EMA-Tent, the average raw gate was 0.4374. Since the base current-batch weight is 0.3, the effective current-batch BN weight became only 0.1312 on average. The median effective current-batch weight was 0.1208, and the 25th percentile was 0.0334.

This confirms that the current gate strongly suppresses current-batch statistics under normal WESAD order.

### Shuffle Order

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

For the default EMA-Tent, the average raw gate was 0.8238. The effective current-batch BN weight became 0.2471 on average, close to the configured maximum of 0.3. The median effective current-batch weight was 0.2592.

This supports the interpretation that current-batch BN statistics are useful when the batch is representative. Tent is stronger than EMA-Tent under shuffle, which means the conservative gate can limit adaptation when the current batch is actually reliable.

## Gate Behavior

The default EMA-Tent gate correlated with true batch class bias:

| Condition | corr(true max class ratio, raw gate) | corr(num present classes, raw gate) |
| --- | ---: | ---: |
| Sequential | -0.4376 | 0.4330 |
| Shuffle | -0.2200 | -0.0015 |

In sequential order, higher label imbalance leads to smaller gate values. This is intended, but in WESAD the imbalance is not an accidental artifact; it is part of the protocol block structure. Therefore, the gate can become overly conservative for long emotional blocks.

## Interpretation

The current EMA-Tent design is effective at preventing the Tent-style collapse seen in sequential WESAD. However, the diagnostic results show that it often achieves this by reducing the current-batch BN contribution substantially. This means the method may be stabilizing adaptation by staying close to source/EMA statistics rather than fully exploiting target-subject information in each block.

The strongest evidence is:

- Sequential Tent collapses: Macro-F1 0.3970.
- Sequential EMA-Tent improves: Macro-F1 0.6136.
- Shuffle Tent is strong: Macro-F1 0.6285.
- Shuffle EMA-Tent is lower than Tent: Macro-F1 0.6111.
- Sequential default EMA-Tent effective batch weight is low: mean 0.1312, median 0.1208.
- Shuffle default EMA-Tent effective batch weight is high: mean 0.2471, median 0.2592.

Thus, the current conclusion should not be "current-batch statistics are bad." A more accurate conclusion is:

> Current-batch statistics are beneficial when a batch is representative, as shown by the shuffle condition. The failure in normal WESAD order arises because consecutive class-biased blocks make current-batch-only normalization unstable. EMA-Tent mitigates this collapse, but the present gate may be overly conservative because it strongly suppresses current-batch statistics in naturally class-biased emotional blocks.

## Next Design Direction

The current default is reasonable as a stabilizing baseline, but not necessarily the best final design. A better version should avoid both extremes:

- Tent: current batch 100%, too aggressive under block order.
- Current EMA-Tent: current batch often too low under block order.

The next candidate should keep a minimum current-batch contribution or use a gate that distinguishes harmful prediction collapse from valid single-emotion blocks. The simple `min_gate` ablation did not clearly improve the average result, so a more targeted block-aware or transition-aware gate is likely more promising than only raising the lower bound.
