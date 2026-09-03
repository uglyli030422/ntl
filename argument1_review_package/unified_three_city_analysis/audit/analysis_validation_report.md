# Analysis Validation Report

## Checks

- Tokyo main results are aggregated from 24 monthly OOF rows to one row per `mesh_code`.
- Tokyo energy-vulnerability metrics are computed from `tokyo_spatial_unit_level_predictions.csv`, not from the 79,104 monthly rows.
- Bias metrics and social vulnerability are z-scored within each city before standardized effect regressions.
- Standardized effects report standardized_beta, cluster_robust_se, 95% CI, p-value, and n_spatial_units.
- Tokyo monthly panel sensitivity includes month fixed effects and mesh_code clustered standard errors.
- Tokyo main vulnerability excludes `apartment_rate`; the old apartment-inclusive index is retained as sensitivity only.
- LinearRegression, RandomForestRegressor, and GradientBoostingRegressor model robustness outputs were regenerated.

## Required Outputs

- `outputs/cross_city/cross_city_standardized_effects.csv`
- `outputs/cross_city/cross_city_bias_metric_robustness.csv`
- `outputs/cross_city/cross_city_model_robustness.csv`
- `outputs/cross_city/tokyo_spatial_unit_level_predictions.csv`
- `figures/result1/cross_city_bias_metric_robustness_forest.png`
- `figures/result1/cross_city_model_robustness_forest.png`

## Output Inventory

- `README.md`
- `audit/analysis_validation_report.md`
- `audit/data_audit_report.csv`
- `audit/data_audit_report.md`
- `audit/vulnerability_definition_report.md`
- `config/city_config.yaml`
- `figures/result1/cross_city_bias_metric_robustness_forest.png`
- `figures/result1/cross_city_model_robustness_forest.png`
- `outputs/amsterdam/city_bias_metrics.csv`
- `outputs/amsterdam/city_energy_vulnerability_metrics.csv`
- `outputs/amsterdam/city_energy_vulnerability_threshold_sensitivity.csv`
- `outputs/amsterdam/city_height_gradient_summary.csv`
- `outputs/amsterdam/city_joint_models.csv`
- `outputs/amsterdam/city_missed_group_comparison.csv`
- `outputs/amsterdam/city_model_metrics.csv`
- `outputs/amsterdam/city_out_of_fold_predictions.csv`
- `outputs/amsterdam/city_social_indicator_effects.csv`
- `outputs/amsterdam/city_spatial_robustness.csv`
- `outputs/amsterdam/city_urban_form_effects.csv`
- `outputs/amsterdam/city_vulnerability_deciles.csv`
- `outputs/cross_city/city_bias_metrics.csv`
- `outputs/cross_city/city_energy_vulnerability_metrics.csv`
- `outputs/cross_city/city_energy_vulnerability_threshold_sensitivity.csv`
- `outputs/cross_city/city_height_gradient_summary.csv`
- `outputs/cross_city/city_joint_models.csv`
- `outputs/cross_city/city_missed_group_comparison.csv`
- `outputs/cross_city/city_model_metrics.csv`
- `outputs/cross_city/city_out_of_fold_predictions.csv`
- `outputs/cross_city/city_social_indicator_effects.csv`
- `outputs/cross_city/city_spatial_robustness.csv`
- `outputs/cross_city/city_urban_form_effects.csv`
- `outputs/cross_city/city_vulnerability_deciles.csv`
- `outputs/cross_city/city_vulnerability_effects.csv`
- `outputs/cross_city/cross_city_bias_metric_robustness.csv`
- `outputs/cross_city/cross_city_indicator_effects.csv`
- `outputs/cross_city/cross_city_model_robustness.csv`
- `outputs/cross_city/cross_city_standardized_effects.csv`
- `outputs/cross_city/cross_city_summary.md`
- `outputs/cross_city/cross_city_vulnerability_effects.csv`
- `outputs/cross_city/tokyo_monthly_panel_fe_sensitivity.csv`
- `outputs/cross_city/tokyo_spatial_unit_level_predictions.csv`
- `outputs/london/city_bias_metrics.csv`
- `outputs/london/city_energy_vulnerability_metrics.csv`
- `outputs/london/city_energy_vulnerability_threshold_sensitivity.csv`
- `outputs/london/city_height_gradient_summary.csv`
- `outputs/london/city_joint_models.csv`
- `outputs/london/city_missed_group_comparison.csv`
- `outputs/london/city_model_metrics.csv`
- `outputs/london/city_out_of_fold_predictions.csv`
- `outputs/london/city_social_indicator_effects.csv`
- `outputs/london/city_spatial_robustness.csv`
- `outputs/london/city_urban_form_effects.csv`
- `outputs/london/city_vulnerability_deciles.csv`
- `outputs/tokyo/city_bias_metrics.csv`
- `outputs/tokyo/city_energy_vulnerability_metrics.csv`
- `outputs/tokyo/city_energy_vulnerability_threshold_sensitivity.csv`
- `outputs/tokyo/city_height_gradient_summary.csv`
- `outputs/tokyo/city_joint_models.csv`
- `outputs/tokyo/city_missed_group_comparison.csv`
- `outputs/tokyo/city_model_metrics.csv`
- `outputs/tokyo/city_out_of_fold_predictions.csv`
- `outputs/tokyo/city_social_indicator_effects.csv`
- `outputs/tokyo/city_spatial_robustness.csv`
- `outputs/tokyo/city_urban_form_effects.csv`
- `outputs/tokyo/city_vulnerability_deciles.csv`
- `scripts/__pycache__/analysis_utils.cpython-311.pyc`
- `scripts/analysis_utils.py`
- `scripts/unified_three_city_analysis.py`
