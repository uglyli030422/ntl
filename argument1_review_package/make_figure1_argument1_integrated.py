from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT / "unified_three_city_analysis"
OUT = PACKAGE_ROOT / "journal_svg_figures" / "Figure1_argument1_integrated.svg"

CITY_ORDER = ["Tokyo", "Amsterdam", "London"]
CITY_COLORS = {"Tokyo": "#0072B2", "Amsterdam": "#D55E00", "London": "#009E73"}
CITY_MARKERS = {"Tokyo": "o", "Amsterdam": "s", "London": "^"}
INK = "#1F2933"
MUTED = "#667085"
GRID = "#D8DEE8"
PALE = "#F4F6F8"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.1,
            "ytick.major.size": 2.1,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def clean(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.40, alpha=0.75)
    ax.set_axisbelow(True)


def label(ax: plt.Axes, letter: str, title: str | None = None, x=-0.11) -> None:
    ax.text(x, 1.08, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")
    if title:
        ax.text(0.0, 1.08, title, transform=ax.transAxes, fontsize=7.1, fontweight="bold", va="top", ha="left")


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def load() -> dict[str, pd.DataFrame]:
    cross_city_pred = ROOT / "outputs/cross_city/city_out_of_fold_predictions.csv"
    if cross_city_pred.exists():
        pred_path = cross_city_pred
        pred = pd.read_csv(
            pred_path,
            usecols=[
                "city",
                "spatial_unit_id",
                "observed_electricity",
                "predicted_electricity",
                "vulnerability_z",
                "standardized_residual",
            ],
            low_memory=False,
        )
    else:
        pred = pd.concat(
            [
                pd.read_csv(
                    ROOT / f"outputs/{city}/city_out_of_fold_predictions.csv",
                    usecols=[
                        "city",
                        "spatial_unit_id",
                        "observed_electricity",
                        "predicted_electricity",
                        "vulnerability_z",
                        "standardized_residual",
                    ],
                    low_memory=False,
                )
                for city in ["tokyo", "amsterdam", "london"]
            ],
            ignore_index=True,
        )
    return {
        "deciles": pd.read_csv(ROOT / "figures/result1/figure1_three_city_standardized_error_deciles.csv"),
        "unit": pd.read_csv(ROOT / "figures/result1/figure1_three_city_standardized_error_unit_values.csv"),
        "effects": pd.read_csv(ROOT / "outputs/cross_city/cross_city_standardized_effects.csv"),
        "metric": pd.read_csv(ROOT / "outputs/cross_city/cross_city_bias_metric_robustness.csv"),
        "model": pd.read_csv(ROOT / "outputs/cross_city/cross_city_model_robustness.csv"),
        "pred": pred,
    }


def pooled_deciles(dec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dcl, g in dec.assign(w=lambda d: d["n"]).groupby("svi_decile"):
        rows.append(
            {
                "svi_decile": dcl,
                "mean": np.average(g["mean_standardized_residual"], weights=g["w"]),
                "lo": np.average(g["ci95_low"], weights=g["w"]),
                "hi": np.average(g["ci95_high"], weights=g["w"]),
                "n": g["n"].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("svi_decile")


def unit_spatial(pred: pd.DataFrame) -> pd.DataFrame:
    return (
        pred.groupby(["city", "spatial_unit_id"], as_index=False)
        .agg(
            observed=("observed_electricity", "mean"),
            predicted=("predicted_electricity", "mean"),
            vulnerability_z=("vulnerability_z", "mean"),
            standardized_residual=("standardized_residual", "mean"),
        )
        .dropna(subset=["observed", "predicted", "vulnerability_z", "standardized_residual"])
    )


def panel_a(ax: plt.Axes, dec: pd.DataFrame) -> None:
    pooled = pooled_deciles(dec)
    for city in CITY_ORDER:
        d = dec[dec.city == city].sort_values("svi_decile")
        ax.plot(d.svi_decile, d.mean_standardized_residual, color=CITY_COLORS[city], alpha=0.42, linewidth=0.9)
        ax.scatter(d.svi_decile, d.mean_standardized_residual, color=CITY_COLORS[city], s=13, marker=CITY_MARKERS[city], alpha=0.62, linewidth=0)
    ax.fill_between(pooled.svi_decile, pooled.lo, pooled.hi, color=INK, alpha=0.10, linewidth=0)
    ax.plot(pooled.svi_decile, pooled["mean"], color=INK, linewidth=2.15, marker="o", markersize=3.6)
    ax.axhline(0, color="#7A869A", linewidth=0.65, linestyle=(0, (3, 2)))
    ax.annotate(
        "systematic overestimation\nin high-SVI units",
        xy=(9.7, pooled.loc[pooled.svi_decile.eq(10), "mean"].iloc[0]),
        xytext=(5.8, 0.67),
        arrowprops={"arrowstyle": "->", "lw": 0.75, "color": INK},
        fontsize=7,
        color=INK,
    )
    ax.set_xlim(1, 10)
    ax.set_ylim(-0.55, 0.84)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlabel("Socioeconomic vulnerability decile")
    ax.set_ylabel("Standardized proxy error")
    clean(ax)
    label(ax, "a", "Shared gradient across 8,725 spatial units")


def panel_b(ax: plt.Axes, unit: pd.DataFrame) -> None:
    rng = np.random.default_rng(3)
    sample_parts = []
    for city, d in unit.groupby("city"):
        n = min(850, len(d))
        sample_parts.append(d.iloc[rng.choice(len(d), n, replace=False)])
    sample = pd.concat(sample_parts)
    hb = ax.hexbin(
        sample.vulnerability_z,
        sample.standardized_residual,
        gridsize=28,
        extent=(-2.5, 2.7, -2.4, 2.8),
        cmap="cividis",
        mincnt=1,
        linewidths=0.0,
        alpha=0.92,
    )
    for city, d in unit.groupby("city"):
        coef = np.polyfit(d.vulnerability_z, d.standardized_residual, 1)
        xs = np.linspace(d.vulnerability_z.quantile(0.04), d.vulnerability_z.quantile(0.96), 50)
        ax.plot(xs, coef[0] * xs + coef[1], color=CITY_COLORS[city], lw=1.0, alpha=0.75)
    ax.axhline(0, color="#7A869A", linewidth=0.6, linestyle=(0, (3, 2)))
    ax.axvline(0, color="#B7C0CC", linewidth=0.55)
    ax.set_xlim(-2.5, 2.7)
    ax.set_ylim(-2.4, 2.8)
    ax.set_xlabel("SVI, z")
    ax.set_ylabel("Proxy error, z")
    clean(ax)
    label(ax, "b", "Unit-level density and city-specific slopes")
    return hb


def panel_c(ax: plt.Axes, effects: pd.DataFrame) -> None:
    d = effects[
        (effects.model_name == "random_forest")
        & (effects.bias_metric == "standardized_residual")
        & (effects.vulnerability_definition == "vulnerability_z")
    ].copy()
    y_base = 0.52
    jit = {"Tokyo": 0.08, "Amsterdam": 0.0, "London": -0.08}
    for row in d.itertuples(index=False):
        y = y_base + jit[row.city]
        ax.plot([row.ci_lower, row.ci_upper], [y, y], color=CITY_COLORS[row.city], linewidth=1.45)
        ax.scatter(row.standardized_beta, y, color=CITY_COLORS[row.city], marker=CITY_MARKERS[row.city], s=28, zorder=3)
        ax.text(row.standardized_beta, y + 0.085, row.city, fontsize=5.8, ha="center", color=CITY_COLORS[row.city])
    ax.axvline(0, color="#7A869A", linewidth=0.65)
    ax.set_ylim(0.25, 0.82)
    ax.set_yticks([])
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("SVI coefficient, beta")
    ax.grid(True, axis="x", color=GRID, linewidth=0.40, alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    label(ax, "c", "All city estimates are positive", x=-0.18)


def panel_d(ax: plt.Axes, pred: pd.DataFrame) -> None:
    u = unit_spatial(pred)
    rows = []
    for city, d in u.groupby("city"):
        d = d.copy()
        d["obs_z"] = zscore(d.observed)
        d["pred_z"] = zscore(d.predicted)
        d["decile"] = pd.qcut(d.observed, 10, labels=False, duplicates="drop") + 1
        g = d.groupby("decile", as_index=False).agg(obs_z=("obs_z", "mean"), pred_z=("pred_z", "mean"))
        g["city"] = city
        rows.append(g)
    curves = pd.concat(rows)
    pooled = curves.groupby("decile", as_index=False).agg(obs_z=("obs_z", "mean"), pred_z=("pred_z", "mean"))
    for city, g in curves.groupby("city"):
        ax.plot(g.decile, g.obs_z, color=CITY_COLORS[city], alpha=0.28, linestyle=(0, (3, 2)), lw=0.8)
        ax.plot(g.decile, g.pred_z, color=CITY_COLORS[city], alpha=0.42, lw=0.9)
    ax.plot(pooled.decile, pooled.obs_z, color=INK, lw=1.55, linestyle=(0, (3, 2)))
    ax.plot(pooled.decile, pooled.pred_z, color=INK, lw=2.0)
    ax.fill_between(pooled.decile, pooled.pred_z, pooled.obs_z, color="#E9B872", alpha=0.20, linewidth=0)
    ax.text(0.05, 0.92, "solid: predicted\ndashed: observed", transform=ax.transAxes, fontsize=6.0, va="top", color=MUTED)
    ax.set_xlim(1, 10)
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.set_xlabel("Observed-electricity decile")
    ax.set_ylabel("Within-city z")
    clean(ax)
    label(ax, "d", "Prediction compression")


def panel_e(ax: plt.Axes, dec: pd.DataFrame) -> None:
    rows = []
    for city, d in dec.groupby("city"):
        low = d[d.svi_decile.isin([1, 2])].mean_standardized_residual.mean()
        high = d[d.svi_decile.isin([9, 10])].mean_standardized_residual.mean()
        rows.append({"city": city, "low": low, "high": high, "gap": high - low})
    q = pd.DataFrame(rows)
    angles = np.deg2rad([105, 0, 255])
    center = np.array([0.0, 0.0])
    radius_low = 0.45
    radius_high = 0.82
    ax.add_patch(plt.Circle(center, radius_low, fill=False, color="#C9D2DD", lw=0.8))
    ax.add_patch(plt.Circle(center, radius_high, fill=False, color="#9AA6B2", lw=0.8, linestyle=(0, (3, 2))))
    for angle, row in zip(angles, q.itertuples(index=False)):
        c = CITY_COLORS[row.city]
        low_xy = np.array([np.cos(angle), np.sin(angle)]) * radius_low
        high_xy = np.array([np.cos(angle), np.sin(angle)]) * radius_high
        ax.plot([low_xy[0], high_xy[0]], [low_xy[1], high_xy[1]], color=c, lw=1.5)
        ax.scatter(*low_xy, s=18, color="#D1D8E2", edgecolor="#7A869A", lw=0.4)
        ax.scatter(*high_xy, s=28, color=c, marker=CITY_MARKERS[row.city], edgecolor=INK, lw=0.35)
        label_xy = high_xy * 1.16
        ax.text(label_xy[0], label_xy[1], f"{row.gap:+.2f}", fontsize=6.2, ha="center", va="center", fontweight="bold", color=c)
    ax.text(0, 0.02, "upper SVI\nquintiles", ha="center", va="center", fontsize=6.1, color=INK)
    ax.text(0, -0.27, "outer ring =\nlarger error", ha="center", va="center", fontsize=5.4, color=MUTED)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.02, 1.02)
    ax.axis("off")
    label(ax, "e", "Quintile shift", x=-0.18)


def panel_f(ax: plt.Axes, metric: pd.DataFrame, model: pd.DataFrame) -> None:
    metric_d = metric[(metric.model_name == "random_forest") & (metric.vulnerability_definition == "vulnerability_z")][
        ["city", "bias_metric", "standardized_beta"]
    ].copy()
    metric_d["test"] = metric_d.bias_metric.map(
        {"relative_bias": "relative", "log_bias": "log", "standardized_residual": "z error", "symmetric_bias": "symmetric"}
    )
    model_d = model[(model.bias_metric == "relative_bias") & (model.vulnerability_definition == "vulnerability_z")][
        ["city", "model_name", "standardized_beta"]
    ].copy()
    model_d["test"] = model_d.model_name.map({"linear": "linear", "random_forest": "forest", "gradient_boosting": "boost"})
    d = pd.concat([metric_d[["city", "test", "standardized_beta"]], model_d[["city", "test", "standardized_beta"]]])
    tests = ["relative", "log", "z error", "symmetric", "linear", "forest", "boost"]
    ypos = np.arange(len(tests))[::-1]
    for yi, test in zip(ypos, tests):
        vals = d[d.test == test]
        xmin, xmax = vals.standardized_beta.min(), vals.standardized_beta.max()
        ax.plot([xmin, xmax], [yi, yi], color="#AAB4C0", lw=1.0)
        for city in CITY_ORDER:
            v = vals[vals.city == city].standardized_beta
            if not v.empty:
                ax.scatter(v.iloc[0], yi, s=16, color=CITY_COLORS[city], marker=CITY_MARKERS[city], alpha=0.9)
    ax.axvline(0, color="#7A869A", lw=0.65)
    ax.set_yticks(ypos)
    ax.set_yticklabels(tests)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("Beta")
    ax.grid(True, axis="x", color=GRID, linewidth=0.40, alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    label(ax, "f", "Robustness envelope")


def panel_g(ax: plt.Axes, pred: pd.DataFrame) -> None:
    u = unit_spatial(pred)
    u["low_obs"] = u.groupby("city")["observed"].transform(lambda s: s <= s.quantile(0.25))
    u["high_svi"] = u.groupby("city")["vulnerability_z"].transform(lambda s: s >= s.quantile(0.75))
    u["high_pred"] = u.groupby("city")["predicted"].transform(lambda s: s >= s.quantile(0.25))
    u["missed"] = u.low_obs & u.high_svi & u.high_pred
    total = int((u.low_obs & u.high_svi).sum())
    missed = int(u.missed.sum())
    share = missed / total if total else np.nan
    colors = ["#E8EDF3", "#D55E00"]
    ax.pie(
        [1 - share, share],
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
    )
    ax.text(0, 0.05, f"{share*100:.0f}%", ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(0, -0.22, "missed among\nlow-electricity,\nhigh-SVI units", ha="center", va="center", fontsize=5.6, color=MUTED)
    label(ax, "g", "Threshold miss", x=-0.18)


def panel_h(ax: plt.Axes, dec: pd.DataFrame) -> None:
    counts = dec.groupby("city", as_index=False).n.sum()
    total = counts.n.sum()
    x = 0
    for row in counts.itertuples(index=False):
        w = row.n / total
        ax.add_patch(Rectangle((x, 0.28), w, 0.26, facecolor=CITY_COLORS[row.city], edgecolor="white", lw=0.7))
        ax.text(x + w / 2, 0.68, f"{row.n:,}", ha="center", va="center", fontsize=6.0, color=INK)
        ax.text(x + w / 2, 0.15, row.city, ha="center", va="center", fontsize=5.7, color=MUTED)
        x += w
    ax.text(0.5, 0.92, f"{total:,} spatial units", ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    label(ax, "h", "Analysis support", x=-0.18)


def main() -> None:
    setup_style()
    data = load()
    unit = unit_spatial(data["pred"])

    fig = plt.figure(figsize=(7.0866, 6.88))
    gs = fig.add_gridspec(
        3,
        4,
        left=0.062,
        right=0.985,
        top=0.955,
        bottom=0.105,
        height_ratios=[1.25, 1.0, 0.76],
        width_ratios=[1.05, 1.05, 0.82, 0.82],
        wspace=0.48,
        hspace=0.62,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[1, 3])
    ax_g = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_note = fig.add_subplot(gs[2, 2:4])

    panel_a(ax_a, data["deciles"])
    panel_b(ax_b, unit)
    panel_c(ax_c, data["effects"])
    panel_d(ax_d, data["pred"])
    panel_e(ax_e, data["deciles"])
    panel_f(ax_f, data["metric"], data["model"])
    panel_g(ax_g, data["pred"])
    panel_h(ax_h, data["deciles"])

    ax_note.axis("off")
    ax_note.add_patch(Rectangle((0.0, 0.10), 1.0, 0.72, facecolor=PALE, edgecolor="#C9D2DD", lw=0.6))
    ax_note.text(0.04, 0.68, "Reading guide", fontsize=7.2, fontweight="bold", color=INK, transform=ax_note.transAxes)
    ax_note.text(
        0.04,
        0.47,
        "The layout treats city as a replication layer rather than a panel axis: black marks show pooled patterns, colored marks show city-level recurrence.",
        fontsize=6.2,
        color=MUTED,
        transform=ax_note.transAxes,
        wrap=True,
    )
    ax_note.text(
        0.04,
        0.24,
        "Proxy error is nighttime-light-predicted electricity minus observed electricity; SVI is standardized within city.",
        fontsize=6.2,
        color=MUTED,
        transform=ax_note.transAxes,
        wrap=True,
    )

    handles = [
        Line2D([0], [0], color=CITY_COLORS[c], marker=CITY_MARKERS[c], linewidth=1.0, markersize=3.6, label=c)
        for c in CITY_ORDER
    ]
    handles.append(Line2D([0], [0], color=INK, marker="o", linewidth=1.8, markersize=3.6, label="pooled"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.062, 0.025), ncol=4, frameon=False, handlelength=1.6, columnspacing=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg")
    print(OUT)


if __name__ == "__main__":
    main()
