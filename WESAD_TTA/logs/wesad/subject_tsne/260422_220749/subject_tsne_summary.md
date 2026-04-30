# WESAD Subject t-SNE Summary

## Setup

- Feature extractor checkpoint: `./ckpt_3class/wesad/cnn1d/S2/wesad_S2_checkpoint.pt`
- Checkpoint fold: `S2`
- Number of subjects: 15
- Number of sampled windows: 1800
- t-SNE perplexity: 30.00
- Max samples per subject and label: 40
- Scatter point size: 40.0

This visualization uses one fixed LOSO-trained 1D-CNN as the feature extractor, so all windows are embedded in the same 64-dimensional feature space before t-SNE.

## Sample Counts

| Subject | baseline | stress | amusement | total |
|---|---:|---:|---:|---:|
| S2 | 40 | 40 | 40 | 120 |
| S3 | 40 | 40 | 40 | 120 |
| S4 | 40 | 40 | 40 | 120 |
| S5 | 40 | 40 | 40 | 120 |
| S6 | 40 | 40 | 40 | 120 |
| S7 | 40 | 40 | 40 | 120 |
| S8 | 40 | 40 | 40 | 120 |
| S9 | 40 | 40 | 40 | 120 |
| S10 | 40 | 40 | 40 | 120 |
| S11 | 40 | 40 | 40 | 120 |
| S13 | 40 | 40 | 40 | 120 |
| S14 | 40 | 40 | 40 | 120 |
| S15 | 40 | 40 | 40 | 120 |
| S16 | 40 | 40 | 40 | 120 |
| S17 | 40 | 40 | 40 | 120 |

## Figures

- Label-wise subject-colored t-SNE: `./logs/wesad/subject_tsne/260422_220749/tsne_by_label_subject_color.png`
- All-window subject-colored t-SNE: `./logs/wesad/subject_tsne/260422_220749/tsne_all_subject_color.png`
- All-window label-colored t-SNE: `./logs/wesad/subject_tsne/260422_220749/tsne_all_label_color.png`

## Interpretation for Paper

If subject-colored clusters remain separated within each emotion-label panel, the visualization supports the claim that physiological features contain strong subject-specific variation even under the same emotion label. This should be treated as qualitative evidence motivating personalized adaptation, not as a standalone quantitative proof.
