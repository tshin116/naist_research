# Related Work Notes

## Tent

Tent performs test-time entropy minimization by updating only the affine
parameters of BatchNorm layers. The proposed MI-gated Dynamic EMA-Tent follows
the same lightweight adaptation target: convolutional and linear layers are
frozen, and only BatchNorm affine parameters are optimized.

## TTN

TTN interpolates source and test BatchNorm statistics. Our method is related in
that it also keeps source statistics as an anchor, but it uses three statistics:
source, target EMA, and current batch. Their weights are controlled by a
batch-level reliability score rather than a fixed source/test interpolation.

## TEMA

TEMA stabilizes test-time BatchNorm by using exponential moving averages of
target statistics. MI-gated Dynamic EMA-Tent also keeps target EMA statistics,
but updates the EMA only when the batch is reliable according to the mutual
information gate.

## MemBN

MemBN uses statistics from previous test batches. Our method uses past target
information through per-BN EMA statistics, avoiding a memory bank and keeping
the implementation lightweight.

## NOTE and RoTTA

NOTE and RoTTA address non-i.i.d. or temporally correlated test streams. This is
close to the WESAD setting, where the original target stream contains block-like
class segments and can produce single-class batches. Our method targets this
failure mode through gate-controlled BatchNorm statistics.

## EATA and SAR

EATA and SAR reduce harmful updates from unreliable or noisy samples. MI-gated
Dynamic EMA-Tent applies a similar idea at the batch level: if the MI gate is
small, entropy minimization is skipped and EMA accumulation is weakened.

## Difference From Prior Work

This project focuses on wearable physiological emotion recognition for unseen
subjects. The main failure mode is not only domain shift, but also WESAD's
ordered block stream, which can create strongly class-biased target batches.

The proposed method separates batch diversity and model uncertainty using a
mutual-information style gate:

- high gate: predictions are class-diverse across the batch and confident per
  sample;
- low gate: the batch is single-class-like, or predictions are globally
  uncertain.

The same gate controls three places: direct current-batch BN statistics, target
EMA accumulation, and entropy-minimization updates. The method does not use a
sample memory bank; it only stores per-BN source statistics and target EMA
statistics.
