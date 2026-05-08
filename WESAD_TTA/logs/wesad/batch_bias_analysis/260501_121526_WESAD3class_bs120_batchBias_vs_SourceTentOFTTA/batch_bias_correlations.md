# Batch Bias Correlations

| target                | bias_metric              | n  | pearson_r | spearman_rho |
| --------------------- | ------------------------ | -- | --------- | ------------ |
| Tent_delta_vs_source  | mean_max_class_ratio     | 15 | 0.3043    | 0.1483       |
| Tent_delta_vs_source  | mean_imbalance           | 15 | 0.2094    | 0.0786       |
| Tent_delta_vs_source  | single_class_batch_ratio | 15 | nan       | nan          |
| Tent_delta_vs_source  | mean_missing_classes     | 15 | nan       | nan          |
| OFTTA_delta_vs_source | mean_max_class_ratio     | 15 | 0.3167    | 0.0912       |
| OFTTA_delta_vs_source | mean_imbalance           | 15 | 0.2237    | 0.0321       |
| OFTTA_delta_vs_source | single_class_batch_ratio | 15 | nan       | nan          |
| OFTTA_delta_vs_source | mean_missing_classes     | 15 | nan       | nan          |
| Tent_shuffle_gain     | mean_max_class_ratio     | 15 | -0.0721   | 0.1680       |
| Tent_shuffle_gain     | mean_imbalance           | 15 | 0.0242    | 0.1964       |
| Tent_shuffle_gain     | single_class_batch_ratio | 15 | nan       | nan          |
| Tent_shuffle_gain     | mean_missing_classes     | 15 | nan       | nan          |
| OFTTA_shuffle_gain    | mean_max_class_ratio     | 15 | -0.3588   | -0.2163      |
| OFTTA_shuffle_gain    | mean_imbalance           | 15 | -0.2557   | -0.1857      |
| OFTTA_shuffle_gain    | single_class_batch_ratio | 15 | nan       | nan          |
| OFTTA_shuffle_gain    | mean_missing_classes     | 15 | nan       | nan          |
