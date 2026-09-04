from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "data" / "result3_multithreshold_luminous_poverty_labels.csv"
OUTDIR = ROOT / "outputs" / "appendix_a2_vulnerability_error_experiments"
OUTDIR.mkdir(parents=True, exist_ok=True)


def qflag(s: pd.Series, q: float, low: bool) -> pd.Series:
    value = s.quantile(q)
    return s.le(value) if low else s.ge(value)


def city_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, g in df.groupby("city", sort=False):
        low_e = qflag(g["observed"], 0.25, low=True)
        high_svi = qflag(g["svi"], 0.75, low=False)
        target = g["actual_vulnerable_low_energy"].astype(bool)
        base_high_svi = high_svi.mean()
        base_low_e = low_e.mean()
        rows.append(
            {
                "city": city,
                "n_units": int(len(g)),
                "low_electricity_units": int(low_e.sum()),
                "high_svi_units": int(high_svi.sum()),
                "vulnerable_low_electricity_units": int(target.sum()),
                "share_high_svi_given_low_electricity": float(high_svi[low_e].mean()),
                "high_svi_baseline_share": float(base_high_svi),
                "high_svi_enrichment_in_low_electricity": float(high_svi[low_e].mean() / base_high_svi),
                "share_low_electricity_given_high_svi": float(low_e[high_svi].mean()),
                "low_electricity_baseline_share": float(base_low_e),
                "low_electricity_enrichment_in_high_svi": float(low_e[high_svi].mean() / base_low_e),
            }
        )
    return pd.DataFrame(rows)


def screening_baselines(df: pd.DataFrame, n_iter: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for city, g in df.groupby("city", sort=False):
        g = g.reset_index(drop=True).copy()
        low_e = qflag(g["observed"], 0.25, low=True).to_numpy()
        high_svi = qflag(g["svi"], 0.75, low=False).to_numpy()
        target = g["actual_vulnerable_low_energy"].astype(bool).to_numpy()
        predicted_low = g["ntl_detected_low_energy"].astype(bool).to_numpy()
        target_n = int(target.sum())
        actual_fn = int((target & ~predicted_low).sum())
        actual_fnr = actual_fn / target_n

        random_fnrs = []
        neutral_fnrs = []
        n_select = int(predicted_low.sum())

        # Observed-electricity quintiles preserve the broad calibration/shrinkage structure.
        q = pd.qcut(g["observed"].rank(method="first"), 5, labels=False).to_numpy()
        residual = (g["proxy_predicted"] - g["observed"]).to_numpy()
        observed = g["observed"].to_numpy()

        for _ in range(n_iter):
            rand_detected = np.zeros(len(g), dtype=bool)
            rand_detected[rng.choice(len(g), size=n_select, replace=False)] = True
            random_fnrs.append(float((target & ~rand_detected).sum() / target_n))

            perm_resid = residual.copy()
            for level in np.unique(q):
                idx = np.where(q == level)[0]
                perm_resid[idx] = rng.permutation(perm_resid[idx])
            neutral_pred = observed + perm_resid
            cutoff = np.quantile(neutral_pred, 0.25)
            neutral_detected = neutral_pred <= cutoff
            neutral_fnrs.append(float((target & ~neutral_detected).sum() / target_n))

        def interval(values: list[float]) -> tuple[float, float, float]:
            arr = np.asarray(values)
            return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

        random_mean, random_low, random_high = interval(random_fnrs)
        neutral_mean, neutral_low, neutral_high = interval(neutral_fnrs)
        rows.append(
            {
                "city": city,
                "n_units": int(len(g)),
                "vulnerable_low_electricity_units": target_n,
                "selected_low_proxy_units": n_select,
                "observed_ntl_false_negative_rate": actual_fnr,
                "observed_ntl_false_negative_count": actual_fn,
                "random_ranking_false_negative_rate_mean": random_mean,
                "random_ranking_false_negative_rate_ci95_low": random_low,
                "random_ranking_false_negative_rate_ci95_high": random_high,
                "neutral_within_electricity_false_negative_rate_mean": neutral_mean,
                "neutral_within_electricity_false_negative_rate_ci95_low": neutral_low,
                "neutral_within_electricity_false_negative_rate_ci95_high": neutral_high,
                "ntl_minus_random_fnr": actual_fnr - random_mean,
                "ntl_minus_neutral_fnr": actual_fnr - neutral_mean,
                "iterations": n_iter,
            }
        )
    return pd.DataFrame(rows)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_summary(coupling: pd.DataFrame, baselines: pd.DataFrame) -> None:
    lines = []
    lines.append("# Appendix A.2 experiment outputs")
    lines.append("")
    lines.append("## Electricity-SVI coupling")
    for _, r in coupling.iterrows():
        lines.append(
            f"- {r.city}: among low-electricity units, {fmt_pct(r.share_high_svi_given_low_electricity)} are high-SVI "
            f"(baseline {fmt_pct(r.high_svi_baseline_share)}, enrichment {r.high_svi_enrichment_in_low_electricity:.2f}x); "
            f"among high-SVI units, {fmt_pct(r.share_low_electricity_given_high_svi)} are low-electricity "
            f"(baseline {fmt_pct(r.low_electricity_baseline_share)}, enrichment {r.low_electricity_enrichment_in_high_svi:.2f}x)."
        )
    lines.append("")
    lines.append("## Screening baselines")
    for _, r in baselines.iterrows():
        lines.append(
            f"- {r.city}: observed NTL missed {fmt_pct(r.observed_ntl_false_negative_rate)} "
            f"({int(r.observed_ntl_false_negative_count)}/{int(r.vulnerable_low_electricity_units)}); "
            f"random ranking baseline {fmt_pct(r.random_ranking_false_negative_rate_mean)} "
            f"[{fmt_pct(r.random_ranking_false_negative_rate_ci95_low)}, {fmt_pct(r.random_ranking_false_negative_rate_ci95_high)}]; "
            f"within-electricity neutral baseline {fmt_pct(r.neutral_within_electricity_false_negative_rate_mean)} "
            f"[{fmt_pct(r.neutral_within_electricity_false_negative_rate_ci95_low)}, {fmt_pct(r.neutral_within_electricity_false_negative_rate_ci95_high)}]."
        )
    (OUTDIR / "appendix_a2_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(INFILE)
    df = df[df["city"].isin(["Amsterdam", "London", "Tokyo"])].copy()
    df = df.rename(columns={"predicted": "proxy_predicted"})
    df = df[
        [
            "city",
            "unit_id",
            "observed",
            "svi",
            "proxy_predicted",
            "actual_vulnerable_low_energy",
            "ntl_detected_low_energy",
        ]
    ].dropna()
    coupling = city_table(df)
    baselines = screening_baselines(df)
    coupling.to_csv(OUTDIR / "table_s3a_electricity_svi_coupling.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(OUTDIR / "table_s3b_screening_baselines.csv", index=False, encoding="utf-8-sig")
    write_summary(coupling, baselines)
    print(coupling.to_string(index=False))
    print()
    print(baselines.to_string(index=False))
    print(OUTDIR)


if __name__ == "__main__":
    main()
