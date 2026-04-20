# Batch Bias Correlations

| target                | bias_metric              | n  | pearson_r | spearman_rho |
| --------------------- | ------------------------ | -- | --------- | ------------ |
| Tent_delta_vs_source  | mean_max_class_ratio     | 15 | -0.7022   | -0.5237      |
| Tent_delta_vs_source  | mean_imbalance           | 15 | -0.5540   | -0.5214      |
| Tent_delta_vs_source  | single_class_batch_ratio | 15 | 0.0174    | 0.0000       |
| Tent_delta_vs_source  | mean_missing_classes     | 15 | 0.0174    | 0.0000       |
| OFTTA_delta_vs_source | mean_max_class_ratio     | 15 | -0.7515   | -0.6434      |
| OFTTA_delta_vs_source | mean_imbalance           | 15 | -0.6373   | -0.6286      |
| OFTTA_delta_vs_source | single_class_batch_ratio | 15 | -0.1382   | -0.1816      |
| OFTTA_delta_vs_source | mean_missing_classes     | 15 | -0.1382   | -0.1816      |
| Tent_shuffle_gain     | mean_max_class_ratio     | 15 | 0.4057    | 0.4933       |
| Tent_shuffle_gain     | mean_imbalance           | 15 | 0.4622    | 0.4929       |
| Tent_shuffle_gain     | single_class_batch_ratio | 15 | 0.2746    | 0.1816       |
| Tent_shuffle_gain     | mean_missing_classes     | 15 | 0.2746    | 0.1816       |
| OFTTA_shuffle_gain    | mean_max_class_ratio     | 15 | 0.4922    | 0.5433       |
| OFTTA_shuffle_gain    | mean_imbalance           | 15 | 0.6202    | 0.5500       |
| OFTTA_shuffle_gain    | single_class_batch_ratio | 15 | 0.6141    | 0.5447       |
| OFTTA_shuffle_gain    | mean_missing_classes     | 15 | 0.6141    | 0.5447       |
