# Core-to-external NTL electricity transfer validation

Training logic: Tokyo, Amsterdam, and London are converted to within-city standardized log NTL and log observed electricity. A single RandomForestRegressor learns observed_z from ntl_z.

Core-city predictions are pooled 5-fold out-of-fold predictions from the same standardized model specification.

External predictions for Marseille and Sydney are direct predictions from the model trained on all three core cities. No Marseille or Sydney observations are used to fit the prediction model.

Residual definition: standardized_residual = predicted_observed_z - observed_z. Positive residual means NTL predicts higher city-relative electricity than observed.

| city      | role            |    n |   spearman_residual_vs_svi |   mean_residual_bottom_svi_decile |   mean_residual_top_svi_decile |   top_minus_bottom_residual |   share_positive_residual_top_svi_decile |
|:----------|:----------------|-----:|---------------------------:|----------------------------------:|-------------------------------:|----------------------------:|-----------------------------------------:|
| Amsterdam | core_oof        |  435 |                   0.536831 |                         -1.19717  |                       0.731636 |                    1.9288   |                                 0.863636 |
| London    | core_oof        | 4994 |                   0.264147 |                         -0.503264 |                       0.403225 |                    0.906489 |                                 0.706    |
| Tokyo     | core_oof        | 3296 |                   0.132207 |                         -0.1515   |                       0.445398 |                    0.596898 |                                 0.654545 |
| Marseille | external_direct |  321 |                   0.486778 |                         -0.6462   |                       0.585046 |                    1.23125  |                                 0.78125  |
| Sydney    | external_direct |  399 |                   0.390145 |                         -0.805762 |                       0.527283 |                    1.33304  |                                 0.7      |