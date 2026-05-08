# Batch Bias Correlations

| target                | bias_metric              | n  | pearson_r | spearman_rho |
| --------------------- | ------------------------ | -- | --------- | ------------ |
| Tent_delta_vs_source  | mean_max_class_ratio     | 15 | 0.1725    | 0.0929       |
| Tent_delta_vs_source  | mean_imbalance           | 15 | 0.1376    | -0.0179      |
| Tent_delta_vs_source  | single_class_batch_ratio | 15 | nan       | nan          |
| Tent_delta_vs_source  | mean_missing_classes     | 15 | nan       | nan          |
| OFTTA_delta_vs_source | mean_max_class_ratio     | 15 | -0.0323   | 0.0697       |
| OFTTA_delta_vs_source | mean_imbalance           | 15 | 0.0582    | 0.1179       |
| OFTTA_delta_vs_source | single_class_batch_ratio | 15 | nan       | nan          |
| OFTTA_delta_vs_source | mean_missing_classes     | 15 | nan       | nan          |
