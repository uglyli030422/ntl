# Cross-City Summary

The corrected main cross-city analysis is run at the spatial-unit level. Tokyo monthly OOF predictions are aggregated to `mesh_code` before cross-city regressions and energy-vulnerability detection.

## Standardized Effects

| city      | model_name    | bias_metric           |   standardized_beta |   cluster_robust_se |   ci_lower |   ci_upper |     p_value |   n_spatial_units |   n_observations | sample_level   | vulnerability_definition   |
|:----------|:--------------|:----------------------|--------------------:|--------------------:|-----------:|-----------:|------------:|------------------:|-----------------:|:---------------|:---------------------------|
| Tokyo     | random_forest | relative_bias         |            0.212668 |           0.0193112 |   0.174818 |   0.250518 | 3.31973e-28 |              3296 |             3296 | spatial_unit   | vulnerability_z            |
| Tokyo     | random_forest | log_bias              |            0.23034  |           0.0174176 |   0.196202 |   0.264479 | 6.33047e-40 |              3296 |             3296 | spatial_unit   | vulnerability_z            |
| Tokyo     | random_forest | standardized_residual |            0.231872 |           0.0181425 |   0.196312 |   0.267431 | 2.10482e-37 |              3296 |             3296 | spatial_unit   | vulnerability_z            |
| Tokyo     | random_forest | symmetric_bias        |            0.230478 |           0.0171206 |   0.196922 |   0.264035 | 2.61688e-41 |              3296 |             3296 | spatial_unit   | vulnerability_z            |
| Amsterdam | random_forest | relative_bias         |            0.466593 |           0.0498703 |   0.368847 |   0.564338 | 8.27232e-21 |               435 |              435 | spatial_unit   | vulnerability_z            |
| Amsterdam | random_forest | log_bias              |            0.46099  |           0.0516278 |   0.359799 |   0.562181 | 4.2949e-19  |               435 |              435 | spatial_unit   | vulnerability_z            |
| Amsterdam | random_forest | standardized_residual |            0.421967 |           0.0586613 |   0.306991 |   0.536943 | 6.32514e-13 |               435 |              435 | spatial_unit   | vulnerability_z            |
| Amsterdam | random_forest | symmetric_bias        |            0.4635   |           0.0509361 |   0.363666 |   0.563335 | 9.06282e-20 |               435 |              435 | spatial_unit   | vulnerability_z            |
| London    | random_forest | relative_bias         |            0.241182 |           0.0147907 |   0.212193 |   0.270172 | 8.89932e-60 |              4994 |             4994 | spatial_unit   | vulnerability_z            |
| London    | random_forest | log_bias              |            0.247879 |           0.014881  |   0.218712 |   0.277045 | 2.6749e-62  |              4994 |             4994 | spatial_unit   | vulnerability_z            |
| London    | random_forest | standardized_residual |            0.246646 |           0.0152529 |   0.21675  |   0.276542 | 8.15065e-59 |              4994 |             4994 | spatial_unit   | vulnerability_z            |
| London    | random_forest | symmetric_bias        |            0.247899 |           0.0148333 |   0.218826 |   0.276972 | 1.06574e-62 |              4994 |             4994 | spatial_unit   | vulnerability_z            |

## Direction Consistency

| bias_metric           | direction_consistent_across_cities   |
|:----------------------|:-------------------------------------|
| log_bias              | True                                 |
| relative_bias         | True                                 |
| standardized_residual | True                                 |
| symmetric_bias        | True                                 |

## Tokyo Monthly Panel Sensitivity

| city   | model_name    | bias_metric           |   standardized_beta |   cluster_robust_se |   ci_lower |   ci_upper |     p_value |   n_spatial_units |   n_observations | sample_level     | vulnerability_definition   |
|:-------|:--------------|:----------------------|--------------------:|--------------------:|-----------:|-----------:|------------:|------------------:|-----------------:|:-----------------|:---------------------------|
| Tokyo  | random_forest | relative_bias         |            0.151685 |           0.0137736 |   0.124688 |   0.178681 | 3.31971e-28 |              3296 |            79104 | monthly_panel_fe | vulnerability_z            |
| Tokyo  | random_forest | log_bias              |            0.17022  |           0.0128715 |   0.144992 |   0.195449 | 6.33042e-40 |              3296 |            79104 | monthly_panel_fe | vulnerability_z            |
| Tokyo  | random_forest | standardized_residual |            0.169264 |           0.0132438 |   0.143306 |   0.195222 | 2.1048e-37  |              3296 |            79104 | monthly_panel_fe | vulnerability_z            |
| Tokyo  | random_forest | symmetric_bias        |            0.169853 |           0.0126172 |   0.145123 |   0.194582 | 2.61686e-41 |              3296 |            79104 | monthly_panel_fe | vulnerability_z            |

## Model Robustness

| city      | model_name        |   standardized_beta |   cluster_robust_se |   ci_lower |   ci_upper |      p_value |   n_spatial_units |         r2 |
|:----------|:------------------|--------------------:|--------------------:|-----------:|-----------:|-------------:|------------------:|-----------:|
| Tokyo     | linear            |            0.20815  |           0.0190698 |   0.170773 |   0.245527 | 9.75397e-28  |              3296 |  0.0109935 |
| Tokyo     | random_forest     |            0.212668 |           0.0193112 |   0.174818 |   0.250518 | 3.31973e-28  |              3296 | -0.0540427 |
| Tokyo     | gradient_boosting |            0.214524 |           0.0193666 |   0.176566 |   0.252483 | 1.62201e-28  |              3296 |  0.0190393 |
| Amsterdam | linear            |            0.565661 |           0.0424766 |   0.482407 |   0.648915 | 1.84359e-40  |               435 |  0.043204  |
| Amsterdam | random_forest     |            0.466593 |           0.0498703 |   0.368847 |   0.564338 | 8.27232e-21  |               435 |  0.214287  |
| Amsterdam | gradient_boosting |            0.466868 |           0.0472175 |   0.374322 |   0.559414 | 4.71123e-23  |               435 |  0.19579   |
| London    | linear            |            0.31635  |           0.0144997 |   0.28793  |   0.344769 | 1.57598e-105 |              4994 |  0.0203538 |
| London    | random_forest     |            0.241182 |           0.0147907 |   0.212193 |   0.270172 | 8.89932e-60  |              4994 | -0.0332943 |
| London    | gradient_boosting |            0.257181 |           0.0149105 |   0.227956 |   0.286405 | 1.15178e-66  |              4994 |  0.0453269 |

## Energy Vulnerability Metrics

| city      |   threshold |   TP |   FP |   FN |   TN |   precision |    recall |        F1 |   specificity |   false_negative_rate |   false_positive_rate |
|:----------|------------:|-----:|-----:|-----:|-----:|------------:|----------:|----------:|--------------:|----------------------:|----------------------:|
| Tokyo     |        0.25 |   67 |  757 |  441 | 2031 |   0.0813107 | 0.13189   | 0.100601  |      0.728479 |              0.86811  |              0.271521 |
| Amsterdam |        0.25 |    3 |  106 |   82 |  244 |   0.0275229 | 0.0352941 | 0.0309278 |      0.697143 |              0.964706 |              0.302857 |
| London    |        0.25 |   43 | 1206 |  796 | 2949 |   0.0344275 | 0.0512515 | 0.0411877 |      0.709747 |              0.948749 |              0.290253 |
