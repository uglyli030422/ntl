# Argument 2 Review Package

This folder contains the code, analysis-ready data, outputs, and figures for Argument 2 in the revised manuscript structure.

## Argument Scope

Argument 2 decomposes the vulnerability-error gradient reported in Argument 1. It does not claim that socioeconomic vulnerability directly causes nighttime-light prediction error. Instead, it tests whether the marginal SVI-error gradient is largely explained by two linked patterns: low observed-electricity units are more susceptible to calibration compression in low-information nighttime-light proxy models, and low observed electricity is socially concentrated in higher-SVI areas.

In the four-argument structure, the logic is:

- Argument 1: Positive nighttime-light electricity-proxy error is concentrated in high-SVI communities.
- Argument 2: Observed-electricity controls and within-electricity strata show that this gradient mainly reflects low-electricity calibration compression combined with low-electricity/SVI co-location.
- Argument 3: Urban form further explains why units with similar observed electricity can have different nighttime-light visibility.
- Argument 4: These error patterns affect screening for vulnerable low-electricity areas.

## Folder Structure

- `scripts/`
  - `build_appendix_a2_experiments.py`: Generates Tables S3a-S3c for electricity-SVI coupling and missed-risk concentration diagnostics.
  - `transfer_validate_core_to_external_cities.py`: Fits the nighttime-light-to-electricity model using Tokyo, Amsterdam, and London, then directly predicts Marseille and Sydney.
  - `make_main_figure2_statistical_compression.py`: Rebuilds the main Argument 2 figure, `Figure2_statistical_compression_mechanism_v1`.
  - `make_main_figure2_statistical_blooming.py`: Archived earlier main Argument 2 figure script, `Figure2_statistical_blooming_v1`.
- `data/`
  - `city_bias_metrics.csv`: Harmonized core-city unit table used for the three-city error analysis.
  - `marseille_fine_scale_bias.csv`: Fine-scale Marseille validation-city input table.
  - `sydney_ausgrid_fine_scale_bias.csv`: Fine-scale Sydney validation-city input table.
  - `result3_multithreshold_luminous_poverty_labels.csv`: Core-city label table used for Appendix A.2 diagnostics.
- `outputs/result2_observed_electricity_controls/`
  - `svi_bias_regression_with_observed_control.csv`: Change in the SVI-error gradient after observed-electricity adjustment.
  - `svi_gradient_within_observed_electricity_quintiles.csv`: City-specific SVI gradients within observed-electricity quintiles.
  - `svi_gradient_pooled_within_observed_quintiles.csv`: Pooled within-quintile SVI-gradient estimates.
  - `calibration_by_observed_electricity_decile.csv`: Calibration summaries by observed-electricity decile.
  - `supplementary_observed_electricity_control_and_calibration.*`: Supplementary observed-electricity control and calibration figure.
- `outputs/appendix_a2_vulnerability_error_experiments/`
  - `table_s3a_electricity_svi_coupling.csv`: Coupling between low observed electricity and high SVI.
  - `table_s3b_screening_baselines.csv`: Missed-risk diagnostics compared with random and within-electricity neutral baselines.
  - `table_s3c_social_concentration_of_missed_low_electricity.csv`: Social concentration of missed low-electricity units.
  - `appendix_a2_experiment_summary.md`: Short text summary of the Appendix A.2 results.
- `outputs/external_transfer_validation_core_to_new_cities/`
  - `core_to_external_transfer_observed_control_table.csv`: Five-city observed-electricity control table, including Marseille and Sydney.
  - `core_to_external_transfer_summary.csv`: Core-city out-of-fold and validation-city direct-transfer residual-SVI summaries.
  - `core_to_external_transfer_svi_deciles.csv`: Residual summaries by SVI decile.
  - `core_to_external_transfer_unit_predictions.csv`: Unit-level core-city and validation-city prediction results.
- `figures/main/`
  - `Figure2_statistical_compression_mechanism_v1.png`
  - `Figure2_statistical_compression_mechanism_v1.svg`
  - `Figure2_statistical_blooming_v1.png`
  - `Figure2_statistical_blooming_v1.svg`
- `figures/supplementary/`
  - `Supplementary_observed_electricity_control_and_calibration.png`
  - `Supplementary_observed_electricity_control_and_calibration.svg`

## Reproduction

Run the scripts from the parent directory of this package:

```bash
python argument2_review_package/scripts/build_appendix_a2_experiments.py
python argument2_review_package/scripts/transfer_validate_core_to_external_cities.py
python argument2_review_package/scripts/make_main_figure2_statistical_compression.py
```

The scripts use package-relative paths and read from the local `data/` and `outputs/` folders. They do not depend on fixed machine-specific paths.

## Marseille and Sydney

Marseille and Sydney are not used for model fitting. In `transfer_validate_core_to_external_cities.py`, their role is `external_direct`: the model is fitted on Tokyo, Amsterdam, and London, and then directly applied to Marseille and Sydney to predict standardized observed electricity and evaluate residual-SVI gradients.

The validation role of these two cities is therefore to test whether the low-electricity compression and residual-SVI gradient patterns appear in held-out cities, not to expand the training set.

## Exclusions

- The removed non-nighttime-light control-model analysis is not included in this package.
- The 147 MB `city_out_of_fold_predictions.csv` table is not included because it exceeds GitHub's 100 MB single-file limit. Argument 2 can be reviewed using the smaller included outputs and unit-level transfer prediction table.
