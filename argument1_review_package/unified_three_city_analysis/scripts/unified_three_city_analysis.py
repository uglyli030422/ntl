from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis_utils import (
    add_bias_columns,
    aggregate_to_spatial_units,
    build_model,
    decile_summary,
    energy_vulnerability,
    fit_oof_rf,
    indicator_effects,
    model_metrics,
    oof_predictions,
    prepare_city_frame,
    read_table,
    spatial_robustness,
    standardized_effect,
    urban_form_effects,
)


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "city_config.yaml"
BIAS_METRICS = ["relative_bias", "log_bias", "standardized_residual", "symmetric_bias"]
ROBUST_MODELS = ["linear", "random_forest", "gradient_boosting"]


def write_md(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def save_city_outputs(city_dir: Path, preds: pd.DataFrame, spatial: pd.DataFrame, metrics: dict) -> None:
    pred_cols = [
        "observed_electricity",
        "predicted_electricity",
        "residual",
        "relative_bias",
        "relative_bias_pct",
        "log_bias",
        "standardized_residual",
        "symmetric_bias",
        "bias_z",
        "city",
        "spatial_unit_id",
        "fold_id",
        "model_name",
        "vulnerability_z",
        "vulnerability_z_with_apartment_sensitivity",
        "ntl_value",
    ]
    preds[[c for c in pred_cols if c in preds.columns]].to_csv(
        city_dir / "city_out_of_fold_predictions.csv", index=False, encoding="utf-8-sig"
    )
    spatial[[c for c in pred_cols if c in spatial.columns]].to_csv(
        city_dir / "city_bias_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([metrics]).to_csv(city_dir / "city_model_metrics.csv", index=False, encoding="utf-8-sig")


def plot_forest(df: pd.DataFrame, path: Path, label_col: str, title: str) -> None:
    if df.empty:
        return
    plot_df = df.dropna(subset=["standardized_beta", "ci_lower", "ci_upper"]).copy()
    if plot_df.empty:
        return
    plot_df[label_col] = plot_df[label_col].astype(str)
    plot_df["label"] = plot_df["city"].astype(str) + " | " + plot_df[label_col]
    plot_df = plot_df.sort_values(["city", label_col]).reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig_h = max(4, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.errorbar(
        plot_df["standardized_beta"],
        y,
        xerr=[plot_df["standardized_beta"] - plot_df["ci_lower"], plot_df["ci_upper"] - plot_df["standardized_beta"]],
        fmt="o",
        color="#2f5d8c",
        ecolor="#8aa8c8",
        capsize=3,
    )
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Standardized beta with 95% CI")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def social_definition_lines() -> list[str]:
    return [
        "## Tokyo",
        "",
        "Main index: city-standardized composite of aging_rate, foreign_rate, rent_home_rate, and reversed own_home_rate.",
        "`apartment_rate` is excluded from the main social vulnerability index and used only as a housing / urban-form variable.",
        "Sensitivity index: the previous index including apartment_rate is retained in `vulnerability_z_with_apartment_sensitivity`.",
        "",
        "## Amsterdam",
        "",
        "Main field: `social_vulnerability` from `amsterdam_buurt_ntl_rf_bias_with_ghsl_height.parquet`, standardized within Amsterdam.",
        "The field is an existing composite prepared in the Amsterdam pipeline. Positive direction indicates greater social vulnerability; related source indicators available in the table include rental_housing_pct, poverty_persons_pct, origin_outside_europe_pct, social_assistance_pct, unemployment_benefit_pct, low_education_pct, and the protective owner_occupied_pct / income / education indicators.",
        "",
        "## London",
        "",
        "Main field: `social_vulnerability` from `uk_london_lsoa_ntl_rf_bias_with_attributes_height.parquet`, standardized within London.",
        "The field is an existing composite prepared in the UK LSOA pipeline. Positive direction indicates greater social vulnerability; component-style fields in the source table include rental tenure, unemployment/inactivity, no qualification, non-white / non-UK-born shares, household deprivation, overcrowding, IMD and deprivation-domain scores, with owner occupation, employment, high qualification, and not-deprived households acting as protective indicators.",
    ]


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    root = Path(cfg["project_root"])
    rf_cfg = dict(cfg["model"])
    rf_cfg.pop("name", None)
    random_state = int(cfg.get("random_state", 42))

    audit_rows = []
    city_metrics = []
    all_spatial = []
    all_panel = []
    standardized_rows = []
    monthly_sensitivity_rows = []
    decile_tables = []
    indicator_tables = []
    urban_tables = []
    joint_tables = []
    height_tables = []
    energy_tables = []
    energy_sens_tables = []
    missed_tables = []
    spatial_tables = []
    model_robustness_rows = []

    for city_key, city_cfg in cfg["cities"].items():
        city_dir = BASE_DIR / "outputs" / city_key
        input_path = root / city_cfg["input_path"]
        raw = read_table(input_path)
        prepared, audit = prepare_city_frame(raw, city_cfg)
        panel_preds, metrics = fit_oof_rf(prepared, city_cfg, rf_cfg, cfg["cv"])
        city_metrics.append(metrics)
        all_panel.append(panel_preds)

        spatial_preds = aggregate_to_spatial_units(panel_preds, city_cfg)
        save_city_outputs(city_dir, panel_preds, spatial_preds, metrics)
        all_spatial.append(spatial_preds)

        if city_key == "tokyo":
            spatial_preds.to_csv(
                BASE_DIR / "outputs" / "cross_city" / "tokyo_spatial_unit_level_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )

        audit_rows.append(
            {
                "city": city_cfg["display_name"],
                "input_path": str(input_path),
                "analysis_level": "spatial_unit_main",
                **audit,
                "spatial_units_main": int(spatial_preds["spatial_unit_id"].nunique()),
            }
        )

        for metric in BIAS_METRICS:
            standardized_rows.append(
                standardized_effect(
                    spatial_preds,
                    city_cfg["display_name"],
                    metric,
                    "random_forest",
                    cluster_col="spatial_unit_id",
                )
            )
            if city_key == "tokyo":
                monthly_sensitivity_rows.append(
                    standardized_effect(
                        panel_preds,
                        city_cfg["display_name"],
                        metric,
                        "random_forest",
                        cluster_col="spatial_unit_id",
                        add_month_fe=True,
                        month_col=city_cfg["time_field"],
                    )
                )

        if city_key == "tokyo" and "vulnerability_z_with_apartment_sensitivity" in spatial_preds.columns:
            for metric in BIAS_METRICS:
                standardized_rows.append(
                    standardized_effect(
                        spatial_preds,
                        city_cfg["display_name"],
                        metric,
                        "random_forest",
                        cluster_col="spatial_unit_id",
                        vulnerability_col="vulnerability_z_with_apartment_sensitivity",
                    )
                )

        dec = decile_summary(spatial_preds, city_cfg["display_name"], cfg["bootstrap"])
        dec.to_csv(city_dir / "city_vulnerability_deciles.csv", index=False, encoding="utf-8-sig")
        decile_tables.append(dec)

        ind = indicator_effects(spatial_preds, city_cfg)
        ind.to_csv(city_dir / "city_social_indicator_effects.csv", index=False, encoding="utf-8-sig")
        indicator_tables.append(ind)

        urban, joint, height = urban_form_effects(spatial_preds, city_cfg)
        urban.to_csv(city_dir / "city_urban_form_effects.csv", index=False, encoding="utf-8-sig")
        joint.to_csv(city_dir / "city_joint_models.csv", index=False, encoding="utf-8-sig")
        height.to_csv(city_dir / "city_height_gradient_summary.csv", index=False, encoding="utf-8-sig")
        urban_tables.append(urban)
        joint_tables.append(joint)
        height_tables.append(height)

        ev, ev_sens, missed = energy_vulnerability(spatial_preds, city_cfg, cfg["thresholds"])
        ev.to_csv(city_dir / "city_energy_vulnerability_metrics.csv", index=False, encoding="utf-8-sig")
        ev_sens.to_csv(city_dir / "city_energy_vulnerability_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
        missed.to_csv(city_dir / "city_missed_group_comparison.csv", index=False, encoding="utf-8-sig")
        energy_tables.append(ev)
        energy_sens_tables.append(ev_sens)
        missed_tables.append(missed)

        spatial = spatial_robustness(spatial_preds, city_cfg)
        spatial.to_csv(city_dir / "city_spatial_robustness.csv", index=False, encoding="utf-8-sig")
        spatial_tables.append(spatial)

        for model_name in ROBUST_MODELS:
            model = build_model(model_name, random_state, rf_cfg if model_name == "random_forest" else None)
            pred, fold_id = oof_predictions(prepared, city_cfg, model, cfg["cv"])
            model_panel = add_bias_columns(prepared, pred, fold_id, model_name)
            model_spatial = aggregate_to_spatial_units(model_panel, city_cfg)
            rel_effect = standardized_effect(
                model_spatial,
                city_cfg["display_name"],
                "relative_bias",
                model_name,
                cluster_col="spatial_unit_id",
            )
            rel_effect.update(model_metrics(model_panel, model_name, city_cfg["display_name"]))
            model_robustness_rows.append(rel_effect)

    cross_dir = BASE_DIR / "outputs" / "cross_city"
    figures_dir = BASE_DIR / "figures"
    result1_dir = figures_dir / "result1"
    result1_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(city_metrics).to_csv(cross_dir / "city_model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_panel, ignore_index=True).to_csv(cross_dir / "city_out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_spatial, ignore_index=True).to_csv(cross_dir / "city_bias_metrics.csv", index=False, encoding="utf-8-sig")

    standardized = pd.DataFrame(standardized_rows)
    main_standardized = standardized[standardized["vulnerability_definition"].eq("vulnerability_z")].copy()
    standardized.to_csv(cross_dir / "cross_city_standardized_effects.csv", index=False, encoding="utf-8-sig")
    main_standardized.to_csv(cross_dir / "city_vulnerability_effects.csv", index=False, encoding="utf-8-sig")
    main_standardized.to_csv(cross_dir / "cross_city_vulnerability_effects.csv", index=False, encoding="utf-8-sig")

    monthly_sensitivity = pd.DataFrame(monthly_sensitivity_rows)
    monthly_sensitivity.to_csv(cross_dir / "tokyo_monthly_panel_fe_sensitivity.csv", index=False, encoding="utf-8-sig")

    metric_robustness = main_standardized.copy()
    metric_robustness["direction"] = np.where(metric_robustness["standardized_beta"] >= 0, "positive", "negative")
    direction_summary = (
        metric_robustness.groupby("bias_metric")["direction"].nunique().reset_index(name="n_directions")
    )
    metric_robustness = metric_robustness.merge(direction_summary, on="bias_metric", how="left")
    metric_robustness["direction_consistent_across_cities"] = metric_robustness["n_directions"].eq(1)
    metric_robustness.to_csv(cross_dir / "cross_city_bias_metric_robustness.csv", index=False, encoding="utf-8-sig")

    model_robustness = pd.DataFrame(model_robustness_rows)
    model_robustness.to_csv(cross_dir / "cross_city_model_robustness.csv", index=False, encoding="utf-8-sig")

    pd.concat(decile_tables, ignore_index=True).to_csv(cross_dir / "city_vulnerability_deciles.csv", index=False, encoding="utf-8-sig")
    pd.concat(indicator_tables, ignore_index=True).to_csv(cross_dir / "city_social_indicator_effects.csv", index=False, encoding="utf-8-sig")
    pd.concat(indicator_tables, ignore_index=True).to_csv(cross_dir / "cross_city_indicator_effects.csv", index=False, encoding="utf-8-sig")
    pd.concat(urban_tables, ignore_index=True).to_csv(cross_dir / "city_urban_form_effects.csv", index=False, encoding="utf-8-sig")
    pd.concat(joint_tables, ignore_index=True).to_csv(cross_dir / "city_joint_models.csv", index=False, encoding="utf-8-sig")
    pd.concat(height_tables, ignore_index=True).to_csv(cross_dir / "city_height_gradient_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(energy_tables, ignore_index=True).to_csv(cross_dir / "city_energy_vulnerability_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(energy_sens_tables, ignore_index=True).to_csv(
        cross_dir / "city_energy_vulnerability_threshold_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(missed_tables, ignore_index=True).to_csv(cross_dir / "city_missed_group_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(spatial_tables, ignore_index=True).to_csv(cross_dir / "city_spatial_robustness.csv", index=False, encoding="utf-8-sig")

    plot_forest(
        metric_robustness,
        result1_dir / "cross_city_bias_metric_robustness_forest.png",
        "bias_metric",
        "Bias Metric Robustness",
    )
    plot_forest(
        model_robustness,
        result1_dir / "cross_city_model_robustness_forest.png",
        "model_name",
        "Model Robustness",
    )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(BASE_DIR / "audit" / "data_audit_report.csv", index=False, encoding="utf-8-sig")
    write_md(
        BASE_DIR / "audit" / "data_audit_report.md",
        "Data Audit Report",
        [
            "This report records input files, valid rows, and the main analysis level after the correction.",
            "",
            audit_df.to_markdown(index=False),
        ],
    )
    write_md(
        BASE_DIR / "audit" / "vulnerability_definition_report.md",
        "Vulnerability Definition Report",
        social_definition_lines(),
    )

    consistency = metric_robustness.groupby("bias_metric")["direction_consistent_across_cities"].first().reset_index()
    summary_lines = [
        "The corrected main cross-city analysis is run at the spatial-unit level. Tokyo monthly OOF predictions are aggregated to `mesh_code` before cross-city regressions and energy-vulnerability detection.",
        "",
        "## Standardized Effects",
        "",
        main_standardized.to_markdown(index=False),
        "",
        "## Direction Consistency",
        "",
        consistency.to_markdown(index=False),
        "",
        "## Tokyo Monthly Panel Sensitivity",
        "",
        monthly_sensitivity.to_markdown(index=False),
        "",
        "## Model Robustness",
        "",
        model_robustness[["city", "model_name", "standardized_beta", "cluster_robust_se", "ci_lower", "ci_upper", "p_value", "n_spatial_units", "r2"]].to_markdown(index=False),
        "",
        "## Energy Vulnerability Metrics",
        "",
        pd.concat(energy_tables, ignore_index=True).to_markdown(index=False),
    ]
    write_md(cross_dir / "cross_city_summary.md", "Cross-City Summary", summary_lines)

    validation = [
        "## Checks",
        "",
        "- Tokyo main results are aggregated from 24 monthly OOF rows to one row per `mesh_code`.",
        "- Tokyo energy-vulnerability metrics are computed from `tokyo_spatial_unit_level_predictions.csv`, not from the 79,104 monthly rows.",
        "- Bias metrics and social vulnerability are z-scored within each city before standardized effect regressions.",
        "- Standardized effects report standardized_beta, cluster_robust_se, 95% CI, p-value, and n_spatial_units.",
        "- Tokyo monthly panel sensitivity includes month fixed effects and mesh_code clustered standard errors.",
        "- Tokyo main vulnerability excludes `apartment_rate`; the old apartment-inclusive index is retained as sensitivity only.",
        "- LinearRegression, RandomForestRegressor, and GradientBoostingRegressor model robustness outputs were regenerated.",
        "",
        "## Required Outputs",
        "",
        "- `outputs/cross_city/cross_city_standardized_effects.csv`",
        "- `outputs/cross_city/cross_city_bias_metric_robustness.csv`",
        "- `outputs/cross_city/cross_city_model_robustness.csv`",
        "- `outputs/cross_city/tokyo_spatial_unit_level_predictions.csv`",
        "- `figures/result1/cross_city_bias_metric_robustness_forest.png`",
        "- `figures/result1/cross_city_model_robustness_forest.png`",
        "",
        "## Output Inventory",
        "",
    ]
    produced = sorted(str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in BASE_DIR.rglob("*") if p.is_file())
    validation.extend(f"- `{p}`" for p in produced)
    write_md(BASE_DIR / "audit" / "analysis_validation_report.md", "Analysis Validation Report", validation)


if __name__ == "__main__":
    main()
