from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def zscore(values: pd.Series) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def zscore_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col in df.columns:
            out[col] = zscore(df[col])
    return out


def make_vulnerability(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, str]:
    field = cfg.get("vulnerability_field")
    if field and field in df.columns:
        return zscore(df[field]), f"Existing field: {field}, city-standardized."

    pos = [c for c in cfg.get("vulnerability_positive", []) if c in df.columns]
    neg = [c for c in cfg.get("vulnerability_negative", []) if c in df.columns]
    parts = []
    for col in pos:
        parts.append(zscore(df[col]))
    for col in neg:
        parts.append(-zscore(df[col]))
    if not parts:
        return pd.Series(np.nan, index=df.index), "No vulnerability field or configured indicators."
    raw = pd.concat(parts, axis=1).mean(axis=1)
    return zscore(raw), f"Composite z-score from positive={pos}, negative={neg}."


def make_configured_index(df: pd.DataFrame, positive: list[str], negative: list[str]) -> pd.Series:
    parts = []
    for col in positive:
        if col in df.columns:
            parts.append(zscore(df[col]))
    for col in negative:
        if col in df.columns:
            parts.append(-zscore(df[col]))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return zscore(pd.concat(parts, axis=1).mean(axis=1))


def prepare_city_frame(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    obs = cfg["observed_electricity"]
    ntl = cfg["ntl_field"]
    sid = cfg["spatial_unit_id"]
    needed = {obs, ntl, sid}
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns for {cfg['display_name']}: {missing}")

    df = raw.copy()
    before = len(df)
    df[obs] = pd.to_numeric(df[obs], errors="coerce")
    df[ntl] = pd.to_numeric(df[ntl], errors="coerce")
    mask = df[obs].notna() & df[ntl].notna()
    if cfg.get("invalid_rules", {}).get("observed_positive", True):
        mask &= df[obs] > 0
    if cfg.get("invalid_rules", {}).get("ntl_nonnegative", True):
        mask &= df[ntl] >= 0
    df = df.loc[mask].copy()
    vuln_z, vuln_note = make_vulnerability(df, cfg)
    df["vulnerability_z"] = vuln_z
    if cfg.get("vulnerability_sensitivity_positive") or cfg.get("vulnerability_sensitivity_negative"):
        df["vulnerability_z_with_apartment_sensitivity"] = make_configured_index(
            df,
            cfg.get("vulnerability_sensitivity_positive", []),
            cfg.get("vulnerability_sensitivity_negative", []),
        )
    df["city"] = cfg["display_name"]
    df["observed_electricity"] = df[obs]
    df["ntl_value"] = df[ntl]
    df["spatial_unit_id"] = df[sid].astype(str)
    audit = {
        "rows_raw": before,
        "rows_valid": len(df),
        "rows_removed": before - len(df),
        "vulnerability_note": vuln_note,
        "observed_nonpositive_raw": int((raw[obs] <= 0).sum()) if obs in raw else None,
        "ntl_negative_raw": int((raw[ntl] < 0).sum()) if ntl in raw else None,
    }
    return df, audit


def build_model(model_name: str, random_state: int, rf_cfg: dict | None = None):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "random_forest":
        base = {
            "n_estimators": 500,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": random_state,
            "n_jobs": -1,
        }
        if rf_cfg:
            base.update(rf_cfg)
            base.pop("name", None)
        return RandomForestRegressor(**base)
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_state, n_estimators=250, min_samples_leaf=5, max_depth=3)
    raise ValueError(f"Unknown model: {model_name}")


def oof_predictions(
    df: pd.DataFrame,
    cfg: dict,
    estimator,
    cv_cfg: dict,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = feature_cols or ["ntl_value"]
    X = df[feature_cols].astype(float)
    y = df["observed_electricity"].astype(float)
    n_splits = min(cv_cfg.get("n_splits", 5), len(df))
    groups = None
    if cfg.get("time_field") and cfg["time_field"] in df.columns and df["spatial_unit_id"].nunique() >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        groups = df["spatial_unit_id"]
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=getattr(estimator, "random_state", 42) or 42)
    pred = np.full(len(df), np.nan)
    fold_id = np.full(len(df), np.nan)
    split_iter = cv.split(X, y, groups=groups) if groups is not None else cv.split(X, y)
    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        fold_model = clone(estimator)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = fold_model.predict(X.iloc[test_idx])
        fold_id[test_idx] = fold
    return pred, fold_id.astype(int)


def add_bias_columns(df: pd.DataFrame, pred: np.ndarray, fold_id: np.ndarray, model_name: str) -> pd.DataFrame:
    out = df.copy()
    out["predicted_electricity"] = pred
    out["residual"] = out["predicted_electricity"] - out["observed_electricity"]
    out["relative_bias"] = out["residual"] / out["observed_electricity"]
    out["relative_bias_pct"] = out["relative_bias"] * 100
    out["log_bias"] = np.where(
        (out["predicted_electricity"] > 0) & (out["observed_electricity"] > 0),
        np.log(out["predicted_electricity"]) - np.log(out["observed_electricity"]),
        np.nan,
    )
    sd_obs = out["observed_electricity"].std(ddof=0)
    out["standardized_residual"] = out["residual"] / sd_obs if sd_obs else np.nan
    denom = out["predicted_electricity"].abs() + out["observed_electricity"].abs()
    out["symmetric_bias"] = np.where(denom > 0, 2 * out["residual"] / denom, np.nan)
    out["bias_z"] = zscore(out["relative_bias"])
    out["fold_id"] = fold_id
    out["model_name"] = model_name
    return out


def model_metrics(out: pd.DataFrame, model_name: str, city: str) -> dict:
    y = out["observed_electricity"].astype(float)
    pred = out["predicted_electricity"].astype(float)
    return {
        "city": city,
        "model_name": model_name,
        "n": len(out),
        "r2": r2_score(y, pred),
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "mean_bias": float(out["residual"].mean()),
        "mean_relative_bias": float(out["relative_bias"].mean()),
        "median_abs_relative_bias": float(out["relative_bias"].abs().median()),
    }


def fit_oof_rf(df: pd.DataFrame, cfg: dict, model_cfg: dict, cv_cfg: dict) -> tuple[pd.DataFrame, dict]:
    rf = build_model("random_forest", model_cfg.get("random_state", 42), model_cfg)
    pred, fold_id = oof_predictions(df, cfg, rf, cv_cfg)
    out = add_bias_columns(df, pred, fold_id, "random_forest")
    return out, model_metrics(out, "random_forest", cfg["display_name"])


def aggregate_to_spatial_units(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not cfg.get("time_field"):
        out = df.copy()
        for metric in ["relative_bias", "log_bias", "standardized_residual", "symmetric_bias"]:
            out[f"{metric}_z"] = zscore(out[metric])
        out["bias_z"] = out["relative_bias_z"]
        out["vulnerability_z"] = zscore(out["vulnerability_z"])
        if "vulnerability_z_with_apartment_sensitivity" in out.columns:
            out["vulnerability_z_with_apartment_sensitivity"] = zscore(out["vulnerability_z_with_apartment_sensitivity"])
        return out

    group = "spatial_unit_id"
    mean_cols = [
        "observed_electricity",
        "predicted_electricity",
        "residual",
        "relative_bias",
        "relative_bias_pct",
        "log_bias",
        "standardized_residual",
        "symmetric_bias",
        "vulnerability_z",
        "vulnerability_z_with_apartment_sensitivity",
        "ntl_value",
    ]
    extra_cols = cfg.get("social_indicators", []) + cfg.get("built_form_indicators", [])
    for col in extra_cols:
        if col not in mean_cols:
            mean_cols.append(col)
    available = [c for c in mean_cols if c in df.columns]
    out = df.groupby(group, as_index=False)[available].mean(numeric_only=True)
    out["city"] = cfg["display_name"]
    out["model_name"] = df["model_name"].iloc[0] if "model_name" in df.columns and len(df) else "unknown"
    for metric in ["relative_bias", "log_bias", "standardized_residual", "symmetric_bias"]:
        out[f"{metric}_z"] = zscore(out[metric])
    out["bias_z"] = out["relative_bias_z"]
    out["vulnerability_z"] = zscore(out["vulnerability_z"])
    if "vulnerability_z_with_apartment_sensitivity" in out.columns:
        out["vulnerability_z_with_apartment_sensitivity"] = zscore(out["vulnerability_z_with_apartment_sensitivity"])
    return out


def standardized_effect(
    df: pd.DataFrame,
    city: str,
    bias_metric: str,
    model_name: str,
    cluster_col: str = "spatial_unit_id",
    vulnerability_col: str = "vulnerability_z",
    add_month_fe: bool = False,
    month_col: str | None = None,
) -> dict:
    cols = [bias_metric, vulnerability_col, cluster_col]
    if add_month_fe and month_col:
        cols.append(month_col)
    use = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(use) < 5 or use[vulnerability_col].nunique() < 2:
        return {
            "city": city,
            "model_name": model_name,
            "bias_metric": bias_metric,
            "standardized_beta": np.nan,
            "cluster_robust_se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_spatial_units": use[cluster_col].nunique() if cluster_col in use else 0,
            "n_observations": len(use),
        }
    use["bias_z_for_model"] = zscore(use[bias_metric])
    use["vulnerability_z_for_model"] = zscore(use[vulnerability_col])
    x_parts = [use[["vulnerability_z_for_model"]]]
    if add_month_fe and month_col:
        dummies = pd.get_dummies(use[month_col].astype(str), prefix="month", drop_first=True, dtype=float)
        x_parts.append(dummies)
    X = sm.add_constant(pd.concat(x_parts, axis=1), has_constant="add").astype(float)
    y = use["bias_z_for_model"].astype(float)
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": use[cluster_col].astype(str)})
    beta = float(fit.params["vulnerability_z_for_model"])
    se = float(fit.bse["vulnerability_z_for_model"])
    return {
        "city": city,
        "model_name": model_name,
        "bias_metric": bias_metric,
        "standardized_beta": beta,
        "cluster_robust_se": se,
        "ci_lower": beta - 1.96 * se,
        "ci_upper": beta + 1.96 * se,
        "p_value": float(fit.pvalues["vulnerability_z_for_model"]),
        "n_spatial_units": int(use[cluster_col].nunique()),
        "n_observations": int(len(use)),
        "sample_level": "monthly_panel_fe" if add_month_fe else "spatial_unit",
        "vulnerability_definition": vulnerability_col,
    }


def ols_result(df: pd.DataFrame, y_col: str, x_cols: list[str], city: str, model_name: str, bias_metric: str) -> dict:
    use = df[[y_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < len(x_cols) + 5:
        return {
            "city": city,
            "model_name": model_name,
            "bias_metric": bias_metric,
            "sample_size": len(use),
            "r2": np.nan,
            "adj_r2": np.nan,
        }
    X = sm.add_constant(use[x_cols], has_constant="add")
    fit = sm.OLS(use[y_col], X).fit()
    out = {
        "city": city,
        "model_name": model_name,
        "bias_metric": bias_metric,
        "sample_size": int(fit.nobs),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
    }
    for col in x_cols:
        out[f"{col}_coefficient"] = float(fit.params.get(col, np.nan))
        out[f"{col}_standard_error"] = float(fit.bse.get(col, np.nan))
        out[f"{col}_ci_lower"] = float(fit.conf_int().loc[col, 0]) if col in fit.params else np.nan
        out[f"{col}_ci_upper"] = float(fit.conf_int().loc[col, 1]) if col in fit.params else np.nan
        out[f"{col}_p_value"] = float(fit.pvalues.get(col, np.nan))
    return out


def bootstrap_mean_ci(values: pd.Series, n_iterations: int, seed: int) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_iterations)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def decile_summary(df: pd.DataFrame, city: str, boot_cfg: dict) -> pd.DataFrame:
    use = df.dropna(subset=["vulnerability_z", "relative_bias"]).copy()
    if use["vulnerability_z"].nunique() < 2:
        return pd.DataFrame()
    use["vulnerability_decile"] = pd.qcut(use["vulnerability_z"], 10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, g in use.groupby("vulnerability_decile"):
        ci_l, ci_u = bootstrap_mean_ci(
            g["relative_bias"], boot_cfg.get("n_iterations", 500), boot_cfg.get("random_state", 42)
        )
        rows.append(
            {
                "city": city,
                "vulnerability_decile": int(decile),
                "n": len(g),
                "vulnerability_z_mean": g["vulnerability_z"].mean(),
                "mean_bias": g["residual"].mean(),
                "median_bias": g["residual"].median(),
                "mean_relative_bias": g["relative_bias"].mean(),
                "median_relative_bias": g["relative_bias"].median(),
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "bias_metric": "relative_bias",
                "model_name": "rf_ntl_only",
            }
        )
    return pd.DataFrame(rows)


def indicator_effects(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    city = cfg["display_name"]
    for col in cfg.get("social_indicators", []):
        if col not in df.columns:
            continue
        tmp = df[["bias_z", col]].copy()
        tmp["indicator_z"] = zscore(tmp[col])
        res = ols_result(tmp, "bias_z", ["indicator_z"], city, "rf_ntl_only", "bias_z")
        rows.append(
            {
                "city": city,
                "indicator_original_name": col,
                "indicator_dimension": "social",
                "coefficient": res.get("indicator_z_coefficient"),
                "standard_error": res.get("indicator_z_standard_error"),
                "ci_lower": res.get("indicator_z_ci_lower"),
                "ci_upper": res.get("indicator_z_ci_upper"),
                "p_value": res.get("indicator_z_p_value"),
                "sample_size": res.get("sample_size"),
                "bias_metric": "bias_z",
                "model_name": "rf_ntl_only",
            }
        )
    return pd.DataFrame(rows)


def urban_form_effects(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city = cfg["display_name"]
    urban_rows = []
    for col in cfg.get("built_form_indicators", []):
        if col in df.columns:
            tmp = df[["vulnerability_z", col, "bias_z"]].copy()
            tmp["urban_form_z"] = zscore(tmp[col])
            rel = ols_result(tmp, "urban_form_z", ["vulnerability_z"], city, col, "urban_form_z")
            bias = ols_result(tmp, "bias_z", ["urban_form_z"], city, col, "bias_z")
            urban_rows.append(
                {
                    "city": city,
                    "urban_form_variable": col,
                    "vulnerability_to_urban_coef": rel.get("vulnerability_z_coefficient"),
                    "vulnerability_to_urban_p": rel.get("vulnerability_z_p_value"),
                    "urban_to_bias_coef": bias.get("urban_form_z_coefficient"),
                    "urban_to_bias_p": bias.get("urban_form_z_p_value"),
                    "sample_size": rel.get("sample_size"),
                }
            )

    x_cols = ["vulnerability_z"]
    urban_z_cols = []
    for col in cfg.get("built_form_indicators", []):
        if col in df.columns:
            z_col = f"z_{col}"
            df[z_col] = zscore(df[col])
            urban_z_cols.append(z_col)
    m1 = ols_result(df, "bias_z", ["vulnerability_z"], city, "M1_bias_vulnerability", "bias_z")
    m2 = ols_result(df, "bias_z", urban_z_cols[:6], city, "M2_bias_urban_form", "bias_z")
    m3 = ols_result(df, "bias_z", x_cols + urban_z_cols[:6], city, "M3_bias_vulnerability_urban", "bias_z")

    height = cfg.get("height_field")
    hsum = pd.DataFrame()
    if height in df.columns:
        tmp = df.dropna(subset=[height, "observed_electricity", "ntl_value"]).copy()
        if len(tmp) >= 10 and tmp[height].nunique() >= 3:
            tmp["height_group"] = pd.qcut(tmp[height], 3, labels=["Low height", "Medium height", "High height"], duplicates="drop")
            tmp["electricity_decile"] = pd.qcut(tmp["observed_electricity"], 10, labels=False, duplicates="drop") + 1
            tmp["ntl_intensity"] = tmp["ntl_value"] / tmp["observed_electricity"]
            hsum = (
                tmp.groupby(["height_group", "electricity_decile"], observed=True)
                .agg(n=("ntl_intensity", "size"), mean_ntl_intensity=("ntl_intensity", "mean"), median_ntl_intensity=("ntl_intensity", "median"))
                .reset_index()
            )
            hsum.insert(0, "city", city)
    return pd.DataFrame(urban_rows), pd.DataFrame([m1, m2, m3]), hsum


def energy_vulnerability(df: pd.DataFrame, cfg: dict, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    sens = []
    comp_rows = []
    city = cfg["display_name"]
    use = df.dropna(subset=["observed_electricity", "ntl_value", "vulnerability_z"]).copy()
    for t in thresholds:
        actual = (use["observed_electricity"] <= use["observed_electricity"].quantile(t)) & (
            use["vulnerability_z"] >= use["vulnerability_z"].median()
        )
        detected = use["ntl_value"] <= use["ntl_value"].quantile(t)
        tp = int((actual & detected).sum())
        fp = int((~actual & detected).sum())
        fn = int((actual & ~detected).sum())
        tn = int((~actual & ~detected).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        row = {
            "city": city,
            "threshold": t,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
            "specificity": specificity,
            "false_negative_rate": fn / (tp + fn) if tp + fn else np.nan,
            "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        }
        sens.append(row)
        if abs(t - 0.25) < 1e-9:
            rows.append(row)
            use["confusion_group"] = np.select(
                [actual & detected, ~actual & detected, actual & ~detected, ~actual & ~detected],
                ["TP", "FP", "FN", "TN"],
            )
            for col in cfg.get("social_indicators", []) + cfg.get("built_form_indicators", []):
                if col in use.columns:
                    tp_vals = pd.to_numeric(use.loc[use["confusion_group"] == "TP", col], errors="coerce").dropna()
                    fn_vals = pd.to_numeric(use.loc[use["confusion_group"] == "FN", col], errors="coerce").dropna()
                    t_p = stats.ttest_ind(tp_vals, fn_vals, equal_var=False, nan_policy="omit").pvalue if len(tp_vals) > 1 and len(fn_vals) > 1 else np.nan
                    u_p = stats.mannwhitneyu(tp_vals, fn_vals, alternative="two-sided").pvalue if len(tp_vals) > 0 and len(fn_vals) > 0 else np.nan
                    comp_rows.append(
                        {
                            "city": city,
                            "variable": col,
                            "tp_mean": tp_vals.mean() if len(tp_vals) else np.nan,
                            "fn_mean": fn_vals.mean() if len(fn_vals) else np.nan,
                            "fn_minus_tp": (fn_vals.mean() - tp_vals.mean()) if len(tp_vals) and len(fn_vals) else np.nan,
                            "welch_t_p_value": t_p,
                            "mannwhitney_u_p_value": u_p,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(sens), pd.DataFrame(comp_rows)


def spatial_robustness(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lon = cfg.get("lon_field")
    lat = cfg.get("lat_field")
    if not lon or not lat or lon not in df.columns or lat not in df.columns:
        return pd.DataFrame(
            [{"city": cfg["display_name"], "model_name": "spatial_quadratic", "status": "no_coordinates", "sample_size": len(df)}]
        )
    tmp = df[["bias_z", "vulnerability_z", lon, lat]].dropna().copy()
    tmp["lon"] = zscore(tmp[lon])
    tmp["lat"] = zscore(tmp[lat])
    tmp["lon2"] = tmp["lon"] ** 2
    tmp["lat2"] = tmp["lat"] ** 2
    tmp["lon_lat"] = tmp["lon"] * tmp["lat"]
    res = ols_result(tmp, "bias_z", ["vulnerability_z", "lon", "lat", "lon2", "lat2", "lon_lat"], cfg["display_name"], "spatial_quadratic", "bias_z")
    res["status"] = "ok"
    return pd.DataFrame([res])
