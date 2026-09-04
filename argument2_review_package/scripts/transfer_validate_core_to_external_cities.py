from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "city_bias_metrics.csv"
SYDNEY = ROOT / "data" / "sydney_ausgrid_fine_scale_bias.csv"
MARSEILLE = ROOT / "data" / "marseille_fine_scale_bias.csv"
OUT = ROOT / "outputs" / "external_transfer_validation_core_to_new_cities"


def zscore(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def spearman(x: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 5:
        return np.nan
    result = spearmanr(d["x"], d["y"])
    return float(getattr(result, "statistic", result[0]))


def standardize_within_city(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ntl = pd.to_numeric(out["ntl"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed = pd.to_numeric(out["observed"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["log_ntl"] = np.log1p(ntl.clip(lower=0))
    out["log_observed"] = np.log1p(observed.clip(lower=0))
    out["ntl_z"] = out.groupby("city", group_keys=False)["log_ntl"].apply(zscore)
    out["observed_z"] = out.groupby("city", group_keys=False)["log_observed"].apply(zscore)
    out["svi_z"] = out.groupby("city", group_keys=False)["svi"].apply(zscore)
    return out


def load_core() -> pd.DataFrame:
    df = pd.read_csv(CORE, low_memory=False)
    cols = ["city", "spatial_unit_id", "observed_electricity", "ntl_value", "vulnerability_z"]
    work = df[cols].rename(
        columns={
            "spatial_unit_id": "unit_id",
            "observed_electricity": "observed",
            "ntl_value": "ntl",
            "vulnerability_z": "svi",
        }
    )
    work["role"] = "core_oof"
    return standardize_within_city(work)


def load_sydney() -> pd.DataFrame:
    df = pd.read_csv(SYDNEY)
    work = pd.DataFrame(
        {
            "city": "Sydney",
            "unit_id": df["sal_code_2021"].astype(str),
            "observed": pd.to_numeric(df["observed_electricity"], errors="coerce"),
            "ntl": pd.to_numeric(df["ntl"], errors="coerce"),
            "svi": pd.to_numeric(df["svi_z"], errors="coerce"),
        }
    )
    work["role"] = "external_direct"
    return standardize_within_city(work)


def load_marseille() -> pd.DataFrame:
    df = pd.read_csv(MARSEILLE)
    work = pd.DataFrame(
        {
            "city": "Marseille",
            "unit_id": df["code_iris"].astype(str),
            "observed": pd.to_numeric(df["observed_electricity"], errors="coerce"),
            "ntl": pd.to_numeric(df["ntl"], errors="coerce"),
            "svi": pd.to_numeric(df["svi_z"], errors="coerce"),
        }
    )
    work["role"] = "external_direct"
    return standardize_within_city(work)


def valid_xy(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["ntl_z", "observed_z", "svi_z"]).copy()


def add_bias_columns(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["predicted_observed_z"] = pred
    out["standardized_residual"] = out["predicted_observed_z"] - out["observed_z"]
    out["residual_z"] = out["standardized_residual"]
    out["svi_decile"] = out.groupby("city", group_keys=False)["svi_z"].apply(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop") + 1
    )
    out["observed_quintile"] = out.groupby("city", group_keys=False)["observed_z"].apply(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop") + 1
    )
    return out


def model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )


def core_oof_predictions(core: pd.DataFrame) -> pd.DataFrame:
    work = valid_xy(core)
    pred = np.full(len(work), np.nan)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    x = work[["ntl_z"]].to_numpy()
    y = work["observed_z"].to_numpy()
    for train_idx, test_idx in cv.split(x, y):
        m = clone(model())
        m.fit(x[train_idx], y[train_idx])
        pred[test_idx] = m.predict(x[test_idx])
    return add_bias_columns(work, pred)


def external_direct_predictions(core: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    train = valid_xy(core)
    test = valid_xy(external)
    m = model()
    m.fit(train[["ntl_z"]], train["observed_z"])
    pred = m.predict(test[["ntl_z"]])
    return add_bias_columns(test, pred)


def summarize(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, g in units.groupby("city"):
        rows.append(
            {
                "city": city,
                "role": g["role"].iloc[0],
                "n": int(len(g)),
                "spearman_residual_vs_svi": spearman(g["standardized_residual"], g["svi_z"]),
                "mean_residual_bottom_svi_decile": float(g.loc[g["svi_decile"].eq(1), "standardized_residual"].mean()),
                "mean_residual_top_svi_decile": float(g.loc[g["svi_decile"].eq(10), "standardized_residual"].mean()),
                "top_minus_bottom_residual": float(
                    g.loc[g["svi_decile"].eq(10), "standardized_residual"].mean()
                    - g.loc[g["svi_decile"].eq(1), "standardized_residual"].mean()
                ),
                "share_positive_residual_top_svi_decile": float(
                    g.loc[g["svi_decile"].eq(10), "standardized_residual"].gt(0).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "city"])


def decile_summary(units: pd.DataFrame) -> pd.DataFrame:
    return (
        units.groupby(["city", "role", "svi_decile"], as_index=False)
        .agg(
            n=("standardized_residual", "size"),
            mean_standardized_residual=("standardized_residual", "mean"),
            median_standardized_residual=("standardized_residual", "median"),
            mean_svi_z=("svi_z", "mean"),
            mean_ntl_z=("ntl_z", "mean"),
            mean_observed_z=("observed_z", "mean"),
        )
        .sort_values(["role", "city", "svi_decile"])
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    core = load_core()
    external = pd.concat([load_marseille(), load_sydney()], ignore_index=True)
    core_pred = core_oof_predictions(core)
    external_pred = external_direct_predictions(core, external)
    units = pd.concat([core_pred, external_pred], ignore_index=True)
    units.to_csv(OUT / "core_to_external_transfer_unit_predictions.csv", index=False, encoding="utf-8-sig")
    summary = summarize(units)
    summary.to_csv(OUT / "core_to_external_transfer_summary.csv", index=False, encoding="utf-8-sig")
    dec = decile_summary(units)
    dec.to_csv(OUT / "core_to_external_transfer_svi_deciles.csv", index=False, encoding="utf-8-sig")
    note = [
        "# Core-to-external NTL electricity transfer validation",
        "",
        "Training logic: Tokyo, Amsterdam, and London are converted to within-city standardized log NTL and log observed electricity. A single RandomForestRegressor learns observed_z from ntl_z.",
        "",
        "Core-city predictions are pooled 5-fold out-of-fold predictions from the same standardized model specification.",
        "",
        "External predictions for Marseille and Sydney are direct predictions from the model trained on all three core cities. No Marseille or Sydney observations are used to fit the prediction model.",
        "",
        "Residual definition: standardized_residual = predicted_observed_z - observed_z. Positive residual means NTL predicts higher city-relative electricity than observed.",
        "",
        summary.to_markdown(index=False),
    ]
    (OUT / "core_to_external_transfer_summary.md").write_text("\n".join(note), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
