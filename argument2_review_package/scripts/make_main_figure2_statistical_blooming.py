from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_DIR = ROOT / "outputs" / "external_transfer_validation_core_to_new_cities"
TRANSFER_UNITS = TRANSFER_DIR / "core_to_external_transfer_unit_predictions.csv"
TRANSFER_DECILES = TRANSFER_DIR / "core_to_external_transfer_svi_deciles.csv"
OUT_EXTERNAL_G_STATS = TRANSFER_DIR / "figure2_external_prediction_error_svi_validation.csv"
OUT_SVG = ROOT / "figures" / "main" / "Figure2_statistical_blooming_v1.svg"
OUT_PNG = ROOT / "figures" / "main" / "Figure2_statistical_blooming_v1.png"

INK = "#20242A"
MUTED = "#687380"
GRID = "#DDE3E8"
BROWN = "#8A4F08"
DARK_BROWN = "#653A05"
TEAL = "#0A5669"
CITY_GREY = "#6F7A86"
ROSE = "#B76E79"
SAGE = "#6E9277"
LAVENDER = "#8D7AAE"
MARKERS = {"Tokyo": "o", "Amsterdam": "s", "London": "^"}
CITY_ORDER = ["Tokyo", "Amsterdam", "London"]
EXTERNAL_MARKERS = {"Marseille": "D", "Sydney": "v"}
EXTERNAL_COLORS = {"Marseille": CITY_GREY, "Sydney": CITY_GREY}
EXTERNAL_REGION_LABELS = {"Marseille IRIS": "Marseille", "Sydney Ausgrid area": "Sydney"}

ERR_CMAP = LinearSegmentedColormap.from_list(
    "teal_champagne_copper",
    ["#063F55", "#3D8491", "#C7DBD8", "#F3ECCD", "#C89435", "#744004"],
    N=256,
)
ERR_NORM = TwoSlopeNorm(vmin=-1.8, vcenter=0, vmax=1.8)
ERROR_BINS = [-np.inf, -1.5, -1.0, -0.6, -0.25, 0.25, 0.6, 1.0, 1.5, np.inf]
ERROR_CENTERS = np.array([-1.8, -1.25, -0.8, -0.42, 0.0, 0.42, 0.8, 1.25, 1.8])
ERROR_LABELS = ["<-1.5", "-1.5", "-1.0", "-0.6", "0", "0.6", "1.0", "1.5", ">1.5"]


def setup() -> None:
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
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )


def panel_letter(ax, letter: str, x: float = -0.10, y: float = 1.08) -> None:
    return None


def blend_with_white(color, density: float):
    rgb = np.array(mpl.colors.to_rgb(color))
    strength = 0.18 + 0.82 * np.clip(density, 0, 1)
    return tuple(1 - strength * (1 - rgb))


def clean(ax, axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=axis, color=GRID, lw=0.38, alpha=0.76)
    ax.set_axisbelow(True)


def shift_axes(axes, dx: float = 0, dy: float = 0, dw: float = 0, dh: float = 0) -> None:
    for ax in axes:
        ax.set_axes_locator(None)
        box = ax.get_position()
        ax.set_position([box.x0 + dx, box.y0 + dy, box.width + dw, box.height + dh])


def add_panel_letters(fig, axes_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g=None) -> None:
    a_box = axes_a[0].get_position()
    b_box = ax_b.get_position()
    c_box = ax_c.get_position()
    d_box = ax_d.get_position()
    e_box = ax_e.get_position()
    f_box = ax_f.get_position()
    top_y = max(a_box.y1, b_box.y1) + 0.010
    letters = {
        "a": (0.055, top_y),
        "b": (b_box.x0 - 0.052, top_y),
        "c": (c_box.x0 - 0.052, c_box.y1 + 0.010),
        "d": (0.055, d_box.y1 + 0.020),
        "e": (e_box.x0 - 0.034, e_box.y1 + 0.028),
        "f": (f_box.x0 - 0.028, f_box.y1 + 0.020),
    }
    if ax_g is not None:
        g_box = ax_g.get_position()
        letters["g"] = (0.055, g_box.y1 + 0.012)
    for letter, (x, y) in letters.items():
        fig.text(x, y, letter, fontsize=8, fontweight="bold", color=INK, ha="left", va="top")


def add_panel_letters_figure1(fig, axes_a, ax_b, ax_c, ax_d, ax_e) -> None:
    a_box = axes_a[0].get_position()
    b_box = ax_b.get_position()
    c_box = ax_c.get_position()
    d_box = ax_d.get_position()
    e_box = ax_e.get_position()
    top_y = max(a_box.y1, b_box.y1) + 0.010
    letters = {
        "a": (0.055, top_y),
        "b": (b_box.x0 - 0.052, top_y),
        "c": (c_box.x0 - 0.052, c_box.y1 + 0.010),
        "d": (0.055, d_box.y1 + 0.020),
        "e": (e_box.x0 - 0.032, e_box.y1 + 0.020),
    }
    for letter, (x, y) in letters.items():
        fig.text(x, y, letter, fontsize=8, fontweight="bold", color=INK, ha="left", va="top")


def load_units() -> pd.DataFrame:
    df = pd.read_csv(TRANSFER_UNITS)
    df = df[
        [
            "city",
            "unit_id",
            "role",
            "observed_z",
            "ntl_z",
            "svi_z",
            "standardized_residual",
            "observed_quintile",
        ]
    ].rename(columns={"observed_z": "observed", "svi_z": "svi"})
    df = df.dropna(subset=["city", "observed", "svi", "standardized_residual"]).copy()
    df["svi_pct"] = df.groupby("city")["svi"].rank(pct=True)
    df["svi_decile"] = np.ceil(df["svi_pct"] * 10).clip(1, 10).astype(int)
    df["error_band"] = pd.cut(df["standardized_residual"], ERROR_BINS, labels=np.arange(1, 10), include_lowest=True).astype(int)
    return df


def combined_units_with_external(units: pd.DataFrame, marseille: pd.DataFrame, sydney: pd.DataFrame) -> pd.DataFrame:
    external = external_validation_data(marseille, sydney).rename(
        columns={"standardized_prediction_error": "standardized_residual", "observed_per_capita": "observed"}
    )
    external["city"] = external["region"].map(EXTERNAL_REGION_LABELS)
    external = external.dropna(subset=["city", "observed", "svi", "standardized_residual"])
    keep = ["city", "observed", "svi", "standardized_residual"]
    out = pd.concat([units[keep], external[keep]], ignore_index=True)
    out["svi_pct"] = out.groupby("city")["svi"].rank(pct=True)
    out["svi_decile"] = np.ceil(out["svi_pct"] * 10).clip(1, 10).astype(int)
    out["error_band"] = pd.cut(out["standardized_residual"], ERROR_BINS, labels=np.arange(1, 10), include_lowest=True).astype(int)
    return out


def panel_a(fig, gs, units: pd.DataFrame) -> list:
    ax_main = fig.add_subplot(gs)
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size="27%", pad=0.020, sharex=ax_main)
    ax_right = divider.append_axes("right", size="29%", pad=0.020, sharey=ax_main)
    ax_left = divider.append_axes("left", size="8.5%", pad=0.045, sharey=ax_main)
    ax_cb = ax_right.inset_axes([0.02, 1.06, 0.94, 0.22])

    counts = units.pivot_table(index="error_band", columns="svi_decile", values="city", aggfunc="count").reindex(index=range(1, 10), columns=range(1, 11)).fillna(0)
    shares = counts.div(counts.sum(axis=0), axis=1) * 100
    max_share = np.nanpercentile(shares.to_numpy(), 97)

    for band in range(1, 10):
        for decile in range(1, 11):
            density = shares.loc[band, decile] / max_share if max_share else 0
            base = ERR_CMAP(ERR_NORM(ERROR_CENTERS[band - 1]))
            face = blend_with_white(base, density)
            ax_main.add_patch(Rectangle((decile - 0.5, band - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=0.38))
            if shares.loc[band, decile] > 0:
                r = 0.06 + 0.28 * np.sqrt(min(shares.loc[band, decile] / max_share, 1))
                ax_main.scatter(decile, band, s=(r * 42) ** 2, facecolor="none", edgecolor=(1, 1, 1, 0.78), lw=0.55, zorder=3)

    ax_main.axhline(5.5, color=INK, lw=0.72)
    ax_main.add_patch(Rectangle((8.1, 6.5), 2.0, 2.0, fill=False, edgecolor=DARK_BROWN, lw=1.05, zorder=4))
    ax_main.text(9.1, 8.64, "high vulnerability", ha="center", va="bottom", fontsize=5.05, color=DARK_BROWN, zorder=5)
    ax_main.set_xlim(0.5, 10.5)
    ax_main.set_ylim(0.5, 9.5)
    ax_main.set_xticks(range(1, 11))
    ax_main.tick_params(axis="x", labelsize=5.4)
    ax_main.set_yticks([1, 3, 5, 7, 9])
    ax_main.set_yticklabels(["-1.5", "-1.0", "0", "1.0", "1.5"])
    ax_main.set_xlabel("Socioeconomic vulnerability decile")
    ax_main.set_ylabel("")
    ax_main.tick_params(length=0)
    ax_main.spines[:].set_visible(False)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_anchor("N")

    by_svi = units.groupby("svi_decile").agg(
        mean_error=("standardized_residual", "mean"),
        over_share=("standardized_residual", lambda x: np.mean(x > 0.25) * 100),
    ).reindex(range(1, 11))
    ax_top.axhline(0, color="#7D8996", lw=0.62)
    ax_top.fill_between(by_svi.index, 0, by_svi.mean_error, where=by_svi.mean_error >= 0, color=BROWN, alpha=0.30, interpolate=True)
    ax_top.fill_between(by_svi.index, 0, by_svi.mean_error, where=by_svi.mean_error < 0, color=TEAL, alpha=0.30, interpolate=True)
    ax_top.plot(by_svi.index, by_svi.mean_error, color=INK, lw=0.92)
    ax_top.set_xlim(0.5, 10.5)
    margin = max(abs(by_svi.mean_error.min()), abs(by_svi.mean_error.max())) * 1.08
    ax_top.set_ylim(-margin, margin)
    ax_top.set_xticks([])
    ax_top.set_yticks([round(-margin / 2, 1), 0, round(margin / 2, 1)])
    ax_top.set_ylabel("mean\nerror", labelpad=2)
    ax_top.spines[["top", "right", "bottom"]].set_visible(False)
    ax_top.grid(True, axis="y", color=GRID, lw=0.35)
    ax_top.tick_params(axis="x", length=0, labelbottom=False)
    panel_letter(ax_top, "a", x=-0.15, y=1.13)

    dist = counts.sum(axis=1)
    dist = dist / dist.sum() * 100
    ax_right.axvline(0, color="#7D8996", lw=0.62)
    for band, value in enumerate(dist.values, start=1):
        face = ERR_CMAP(ERR_NORM(ERROR_CENTERS[band - 1]))
        ax_right.barh(band, value, color=face, edgecolor="white", lw=0.38, height=0.68, alpha=0.86)
    ax_right.set_ylim(0.5, 9.5)
    ax_right.set_xlim(0, max(25, dist.max() * 1.15))
    ax_right.set_yticks([])
    ax_right.set_xticks([0, 10, 20])
    ax_right.set_xlabel("share\nof units (%)", labelpad=2)
    ax_right.spines[["top", "right", "left"]].set_visible(False)
    ax_right.grid(True, axis="x", color=GRID, lw=0.35)

    ax_left.axis("off")
    for i in range(96):
        v = -1.8 + 3.6 * (i + 0.5) / 96
        ax_left.add_patch(
            Rectangle(
                (0.42, i / 96),
                0.20,
                1 / 96,
                transform=ax_left.transAxes,
                facecolor=ERR_CMAP(ERR_NORM(v)),
                edgecolor="none",
            )
        )
    ax_left.text(0.52, 1.025, "+", transform=ax_left.transAxes, fontsize=7.2, color=MUTED, ha="center", va="bottom")
    ax_left.text(0.52, 0.50, "0", transform=ax_left.transAxes, fontsize=5.8, color=MUTED, ha="center", va="center")
    ax_left.text(0.52, -0.035, "-", transform=ax_left.transAxes, fontsize=7.2, color=MUTED, ha="center", va="top")
    ax_left.text(
        -0.95,
        0.50,
        "Standardized proxy-error band",
        transform=ax_left.transAxes,
        rotation=90,
        fontsize=7,
        color=INK,
        ha="center",
        va="center",
    )

    ax_cb.axis("off")
    for i, (pct, xpos) in enumerate(zip([5, 10, 20], [0.16, 0.48, 0.82])):
        r = 0.06 + 0.28 * np.sqrt(pct / max_share)
        ax_cb.scatter(xpos, 0.72, s=(r * 42) ** 2, facecolor="none", edgecolor=CITY_GREY, lw=0.65, transform=ax_cb.transAxes)
        ax_cb.text(xpos, 0.36, f"{pct}%", transform=ax_cb.transAxes, ha="center", va="center", fontsize=5.3, color=MUTED)
    ax_cb.text(0.50, 0.08, "within-SVI share", transform=ax_cb.transAxes, fontsize=5.1, color=MUTED, ha="center", va="bottom")
    return [ax_top, ax_main, ax_right, ax_left, ax_cb]


def panel_b(ax, dec: pd.DataFrame) -> None:
    plot_dec = dec.copy()
    if "ci95_low" not in plot_dec.columns:
        plot_dec["ci95_low"] = plot_dec["mean_standardized_residual"]
        plot_dec["ci95_high"] = plot_dec["mean_standardized_residual"]
    rows = []
    for d, g in plot_dec.groupby("svi_decile"):
        w = g.n.to_numpy()
        rows.append((d, np.average(g.mean_standardized_residual, weights=w), np.average(g.ci95_low, weights=w), np.average(g.ci95_high, weights=w)))
    pooled = pd.DataFrame(rows, columns=["decile", "mean", "lo", "hi"]).sort_values("decile")
    for city in CITY_ORDER + ["Marseille", "Sydney"]:
        d = plot_dec[plot_dec.city.eq(city)].sort_values("svi_decile")
        marker = MARKERS.get(city, EXTERNAL_MARKERS.get(city, "o"))
        ax.plot(d.svi_decile, d.mean_standardized_residual, color="#B3BDC7", lw=0.85, alpha=0.72)
        ax.scatter(d.svi_decile, d.mean_standardized_residual, marker=marker, facecolor="white", edgecolor=CITY_GREY, s=13, lw=0.65, zorder=3)
    ax.fill_between(pooled.decile, pooled.lo, pooled.hi, color=BROWN, alpha=0.16, lw=0)
    ax.fill_between(pooled.decile, 0, pooled["mean"], where=pooled["mean"] >= 0, color=BROWN, alpha=0.10, lw=0)
    ax.fill_between(pooled.decile, 0, pooled["mean"], where=pooled["mean"] < 0, color=TEAL, alpha=0.10, lw=0)
    ax.plot(pooled.decile, pooled["mean"], color=BROWN, lw=2.1, marker="o", markersize=3.4)

    ax.axhline(0, color="#7D8996", lw=0.65, linestyle=(0, (3, 2)))
    ax.set_xlim(0.85, 10.35)
    ax.set_ylim(-1.02, 1.10)
    ax.set_xticks(range(1, 11))
    ax.set_yticks([-1.0, -0.5, 0, 0.5])
    ax.tick_params(axis="x", labelsize=5.4)
    ax.set_xlabel("SVI decile")
    ax.set_ylabel("standardized error")
    clean(ax)
    panel_letter(ax, "b")


def smooth_hist(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    kernel = np.array([1, 2, 4, 6, 4, 2, 1], dtype=float)
    kernel /= kernel.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, np.convolve(hist, kernel, mode="same")


def panel_c(ax, units: pd.DataFrame) -> None:
    df = units.copy()
    df["svi_pct"] = df.groupby("city")["svi"].rank(pct=True)
    groups = [
        ("bottom 20%", df.svi_pct <= 0.20, TEAL),
        ("middle 60%", df.svi_pct.between(0.20, 0.80), "#D8C58B"),
        ("top 20%", df.svi_pct >= 0.80, BROWN),
    ]
    bins = np.linspace(-2.8, 2.8, 45)
    offsets = [0.0, 0.56, 1.12]
    for offset, (label, mask, color) in zip(offsets, groups):
        x, y = smooth_hist(df.loc[mask, "standardized_residual"].to_numpy(), bins)
        y = y / y.max() * 0.40
        ax.fill_between(x, offset, offset + y, color=color, alpha=0.45, lw=0)
        ax.plot(x, offset + y, color=color, lw=1.05)
        med = df.loc[mask, "standardized_residual"].median()
        ax.scatter(med, offset, s=18, marker="D", facecolor=color, edgecolor=INK, lw=0.35, zorder=4)
        ax.text(-2.70, offset + 0.08, label, fontsize=5.8, color=INK, ha="left", va="bottom")
    ax.axvline(0, color="#7D8996", lw=0.65, linestyle=(0, (3, 2)))
    ax.text(-1.95, 1.60, "under", fontsize=5.7, color=TEAL, ha="center")
    ax.text(1.95, 1.60, "over", fontsize=5.7, color=BROWN, ha="center")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-0.08, 1.68)
    ax.set_yticks([])
    ax.set_xlabel("standardized proxy error", labelpad=2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.38)
    panel_letter(ax, "c")


def external_validation_data(marseille: pd.DataFrame, sydney: pd.DataFrame) -> pd.DataFrame:
    frames = [
        prepare_external_validation(marseille, "Marseille IRIS", "IRIS"),
        prepare_external_validation(sydney, "Sydney Ausgrid area", "SALs"),
    ]
    return pd.concat(frames, ignore_index=True)


def panel_d(ax, units: pd.DataFrame) -> None:
    q_colors = ["#7AA6B2", "#BFD7D5", "#EFE7CB", "#DCB76C", "#9A6215"]
    rows = []
    for city, d in units.dropna(subset=["observed_quintile"]).groupby("city"):
        for q, g in d.groupby("observed_quintile"):
            center, lo, hi = bootstrap_slope_interval(
                g["svi"].to_numpy(),
                g["standardized_residual"].to_numpy(),
                seed=100 + int(q) + 7 * (CITY_ORDER + ["Marseille", "Sydney"]).index(city),
                draws=250,
            )
            rows.append({"city": city, "q": int(q), "coef": center, "lo": lo, "hi": hi, "n": len(g)})
    quint = pd.DataFrame(rows)
    pooled = []
    for q, g in quint.groupby("q"):
        weights = g.n.to_numpy()
        pooled.append((int(q), np.average(g.coef, weights=weights), np.average(g.lo, weights=weights), np.average(g.hi, weights=weights)))
    pooled = pd.DataFrame(pooled, columns=["q", "coef", "lo", "hi"]).sort_values("q")

    for q in range(1, 6):
        ax.axvspan(q - 0.45, q + 0.45, color=q_colors[q - 1], alpha=0.13, lw=0)
    offsets = {"Tokyo": -0.16, "Amsterdam": -0.08, "London": 0.00, "Marseille": 0.08, "Sydney": 0.16}
    for city in CITY_ORDER + ["Marseille", "Sydney"]:
        d = quint[quint.city.eq(city)].sort_values("q")
        ax.scatter(
            d.q + offsets[city],
            d.coef,
            marker=MARKERS.get(city, EXTERNAL_MARKERS.get(city, "o")),
            s=17,
            facecolor="white",
            edgecolor=CITY_GREY,
            lw=0.62,
            zorder=3,
        )
    ax.vlines(pooled.q, pooled.lo, pooled.hi, color=INK, lw=0.75, zorder=4)
    ax.scatter(pooled.q, pooled.coef, marker="D", s=28, facecolor=BROWN, edgecolor=INK, lw=0.45, zorder=5)
    ax.axhline(0, color="#7D8996", lw=0.65, linestyle=(0, (3, 2)))
    ax.set_xlim(0.55, 5.55)
    ax.set_ylim(-0.42, 0.62)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Observed-use quintile")
    ax.set_ylabel("within-strata SVI coefficient")
    clean(ax)
    panel_letter(ax, "d", x=-0.15, y=1.10)


def ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.26) -> None:
    dx = (x1 - x0) * 0.52
    verts = [
        (x0, y0a),
        (x0 + dx, y0a),
        (x1 - dx, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - dx, y1b),
        (x0 + dx, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def panel_e(ax, units: pd.DataFrame) -> None:
    df = units.copy()
    df["svi_pct"] = df.groupby("city")["svi"].rank(pct=True)
    df["positive_exposure"] = (df.standardized_residual > 0.25).astype(float)
    ranked = df.sort_values("svi_pct").reset_index(drop=True)
    x = np.r_[0, np.arange(1, len(ranked) + 1) / len(ranked)]
    y = np.r_[0, ranked.positive_exposure.cumsum().to_numpy() / ranked.positive_exposure.sum()]
    y80 = np.interp(0.80, x, y)
    burden = (1 - y80) * 100
    enrichment = burden / 20

    ax.plot([0, 1], [0, 1], color="#C8D0D6", lw=0.62, linestyle=(0, (3, 2)))
    ax.fill_between(x, x, y, where=y < x, color="#D9E8E7", alpha=0.72, lw=0)
    ax.fill_between(x, x, y, where=x >= 0.80, color="#E6C178", alpha=0.30, lw=0)
    ax.plot(x, y, color=INK, lw=1.05)
    ax.plot([0.80, 0.80], [0, y80], color="#7D8996", lw=0.58, linestyle=(0, (2, 2)))
    ax.plot([0.80, 1.00], [y80, y80], color=BROWN, lw=1.15)
    ax.scatter([0.80, 1.00], [y80, 1.00], s=[14, 18], facecolor=[TEAL, BROWN], edgecolor=INK, lw=0.38, zorder=4)
    ax.annotate(
        f"highest SVI quintile\ncaptures {burden:.0f}% of positive error",
        xy=(0.90, (1 + y80) / 2),
        xytext=(0.43, 0.86),
        ha="center",
        va="center",
        fontsize=5.25,
        color=DARK_BROWN,
        arrowprops={"arrowstyle": "-", "lw": 0.55, "color": DARK_BROWN, "shrinkA": 1, "shrinkB": 2},
    )
    ax.text(0.07, 0.84, f"{enrichment:.1f}x", fontsize=7.2, color=INK, ha="left", va="center")
    ax.text(0.07, 0.76, "burden", fontsize=5.1, color=MUTED, ha="left", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_xlabel("cumulative units ranked by SVI (%)")
    ax.set_ylabel("positive-error\nexposure (%)", labelpad=-5)
    clean(ax)
    panel_letter(ax, "e", x=-0.14)


def sensitivity_category(name: str) -> str | None:
    n = name.lower()
    if "income" in n or "owner" in n or "own_home" in n:
        return None
    if "rent" in n or "rental" in n:
        return "rental"
    if "foreign" in n or "origin" in n or "non_uk" in n or "non_white" in n:
        return "migration"
    if "aging" in n or "age" in n:
        return "ageing"
    if "unemployment" in n or "employment" in n or "inactive" in n or "labor" in n or "assistance" in n:
        return "work"
    if "education" in n or "qualification" in n or "graduate" in n:
        return "education"
    if "occupation" in n:
        return "occupation"
    if "depriv" in n or "poverty" in n or "overcrowd" in n or "living" in n or "health" in n:
        return "deprivation"
    return None


def bootstrap_slope_interval(x: np.ndarray, y: np.ndarray, seed: int = 7, draws: int = 400) -> tuple[float, float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[ok]
    y = np.asarray(y)[ok]
    center = float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 else np.nan
    if len(x) < 6:
        return center, center, center
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, len(x), len(x))
        if np.unique(x[idx]).size < 3:
            continue
        vals.append(np.polyfit(x[idx], y[idx], 1)[0])
    if not vals:
        return center, center, center
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return center, float(lo), float(hi)


def bootstrap_slope_samples(x: np.ndarray, y: np.ndarray, seed: int = 7, draws: int = 11) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[ok]
    y = np.asarray(y)[ok]
    if len(x) < 6:
        return np.array([])
    rng = np.random.default_rng(seed)
    vals = []
    attempts = 0
    while len(vals) < draws and attempts < draws * 12:
        attempts += 1
        idx = rng.integers(0, len(x), len(x))
        if np.unique(x[idx]).size < 3:
            continue
        vals.append(np.polyfit(x[idx], y[idx], 1)[0])
    return np.asarray(vals, dtype=float)


def panel_f(ax, units: pd.DataFrame) -> None:
    city_colors = {
        "Tokyo": TEAL,
        "Amsterdam": "#74AEB6",
        "London": "#D8C58B",
        "Marseille": "#C08A34",
        "Sydney": BROWN,
    }
    y_positions = {"Tokyo": 4.65, "Amsterdam": 3.55, "London": 2.45, "Marseille": 1.35, "Sydney": 0.35}
    for i, city in enumerate(CITY_ORDER + ["Marseille", "Sydney"]):
        d = units[units.city.eq(city)]
        center, lo, hi = bootstrap_slope_interval(
            d["svi"].to_numpy(),
            d["standardized_residual"].to_numpy(),
            seed=21 + i,
        )
        samples = bootstrap_slope_samples(
            d["svi"].to_numpy(),
            d["standardized_residual"].to_numpy(),
            seed=41 + i,
            draws=15,
        )
        y0 = y_positions[city]
        color = city_colors[city]
        if len(samples):
            bins = np.linspace(max(-0.05, np.nanmin(samples) - 0.08), min(0.85, np.nanmax(samples) + 0.08), 42)
            x, dens = smooth_hist(samples, bins)
            if dens.max() > 0:
                dens = dens / dens.max() * 0.18
                ax.fill_between(x, y0, y0 + dens, color=color, alpha=0.18, lw=0, zorder=1)
                ax.plot(x, y0 + dens, color=color, lw=0.72, zorder=2)
        ax.plot([lo, hi], [y0 - 0.075, y0 - 0.075], color=color, lw=1.15, solid_capstyle="round", zorder=2)
        ax.plot([lo, lo], [y0 - 0.115, y0 - 0.035], color=color, lw=0.7, zorder=2)
        ax.plot([hi, hi], [y0 - 0.115, y0 - 0.035], color=color, lw=0.7, zorder=2)
        for j, val in enumerate(samples):
            jitter = -0.10 - (j % 4) * 0.038
            ax.scatter(val, y0 + jitter, s=15, marker=MARKERS.get(city, EXTERNAL_MARKERS.get(city, "o")),
                       facecolor="white", edgecolor=color, lw=0.58, zorder=3)
        ax.scatter(center, y0 - 0.075, marker="D", s=30, facecolor=color, edgecolor=INK, lw=0.45, zorder=4)
        ax.text(-0.022, y0, city, ha="right", va="center", fontsize=5.8, color=INK)
    ax.axvline(0, color="#7D8996", lw=0.62, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.035, 0.85)
    ax.set_ylim(0.00, 5.05)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("SVI coefficient")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.35)
    ax.tick_params(axis="x", labelsize=5.5)
    panel_letter(ax, "f", x=-0.18, y=1.08)


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3:
        return 0.0, float(np.nanmean(y)), np.nan
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    return slope, intercept, corr


def prepare_external_validation(df: pd.DataFrame, region: str, unit_label: str) -> pd.DataFrame:
    d = df.rename(
        columns={
            "svi_z": "svi",
            "ntl_prediction_residual": "prediction_error",
        }
    ).replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction_error", "svi"]).copy()
    sd = d["prediction_error"].std(ddof=0)
    d["standardized_prediction_error"] = (d["prediction_error"] - d["prediction_error"].mean()) / sd
    d["svi_quintile"] = pd.qcut(d["svi"].rank(method="first"), 5, labels=False) + 1
    d["region"] = region
    d["unit_label"] = unit_label
    keep = ["region", "unit_label", "svi_quintile", "svi", "prediction_error", "standardized_prediction_error"]
    if "observed_per_capita" in d.columns:
        keep.append("observed_per_capita")
    return d[keep]


def save_external_validation_stats(data: pd.DataFrame) -> None:
    rows = []
    for region, d in data.groupby("region"):
        rho = d["svi"].corr(d["prediction_error"], method="spearman")
        by_q = d.groupby("svi_quintile")["standardized_prediction_error"].agg(["count", "mean", "std"]).reset_index()
        q1 = by_q.loc[by_q.svi_quintile.eq(1), "mean"].iloc[0]
        q5 = by_q.loc[by_q.svi_quintile.eq(5), "mean"].iloc[0]
        for _, row in by_q.iterrows():
            n = int(row["count"])
            se = row["std"] / np.sqrt(n) if n > 1 else 0
            rows.append(
                {
                    "region": region,
                    "svi_quintile": int(row["svi_quintile"]),
                    "n": n,
                    "mean_standardized_prediction_error": row["mean"],
                    "ci95_low": row["mean"] - 1.96 * se,
                    "ci95_high": row["mean"] + 1.96 * se,
                    "spearman_prediction_error_vs_svi": rho,
                    "q5_minus_q1_standardized_error": q5 - q1,
                }
            )
    pd.DataFrame(rows).to_csv(OUT_EXTERNAL_G_STATS, index=False)


def panel_g(ax, marseille: pd.DataFrame, sydney: pd.DataFrame) -> None:
    frames = [
        prepare_external_validation(marseille, "Marseille IRIS", "IRIS"),
        prepare_external_validation(sydney, "Sydney Ausgrid area", "SALs"),
    ]
    data = pd.concat(frames, ignore_index=True)
    save_external_validation_stats(data)
    colors = {"Marseille IRIS": TEAL, "Sydney Ausgrid area": BROWN}
    offsets = {"Marseille IRIS": -0.045, "Sydney Ausgrid area": 0.045}
    ax.axhline(0, color="#7D8996", lw=0.65, linestyle=(0, (3, 2)))
    for region in ["Marseille IRIS", "Sydney Ausgrid area"]:
        d = data[data.region.eq(region)]
        stats = []
        for q, g in d.groupby("svi_quintile"):
            mean = g.standardized_prediction_error.mean()
            se = g.standardized_prediction_error.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0
            stats.append((int(q), mean, mean - 1.96 * se, mean + 1.96 * se, len(g)))
        stats = pd.DataFrame(stats, columns=["q", "mean", "lo", "hi", "n"]).sort_values("q")
        color = colors[region]
        off = offsets[region]
        for q, g in d.groupby("svi_quintile"):
            jitter = np.linspace(-0.12, 0.12, len(g)) if len(g) > 1 else np.array([0.0])
            y = np.clip(g.standardized_prediction_error.to_numpy(), -2.3, 2.3)
            ax.scatter(np.full(len(g), q + off) + jitter * 0.42, y, s=8, facecolor=color, edgecolor="none", alpha=0.17, zorder=2)
        xs = stats.q.to_numpy(dtype=float) + off
        ys = stats["mean"].to_numpy(dtype=float)
        degree = 2 if len(xs) >= 3 else 1
        coef = np.polyfit(xs, ys, degree)
        xx = np.linspace(xs.min(), xs.max(), 100)
        label = region.replace(" IRIS", "").replace(" Ausgrid area", "")
        ax.plot(xx, np.polyval(coef, xx), color=color, lw=1.65, zorder=4, label=label)
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 1.08), frameon=False, fontsize=5.5, handlelength=1.9, borderaxespad=0.0, ncol=2, columnspacing=1.1)
    ax.set_xlim(0.55, 5.50)
    ax.set_ylim(-2.35, 2.35)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("SVI quintile in external region")
    ax.set_ylabel("standardized\nprediction error")
    ax.grid(True, axis="y", color=GRID, lw=0.35, alpha=0.66)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    setup()
    units_all = load_units()
    dec = pd.read_csv(TRANSFER_DECILES)

    fig = plt.figure(figsize=(7.0866, 6.10))
    gs = fig.add_gridspec(
        3,
        6,
        left=0.070,
        right=0.955,
        top=0.935,
        bottom=0.075,
        width_ratios=[1.24, 1.24, 1.24, 0.84, 0.84, 0.84],
        height_ratios=[1.12, 0.88, 0.92],
        wspace=0.60,
        hspace=0.56,
    )
    axes_a = panel_a(fig, gs[0:2, 0:3], units_all)
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c = fig.add_subplot(gs[1, 3:6])
    ax_d = fig.add_subplot(gs[2, 0:3])
    ax_e = fig.add_subplot(gs[2, 3:6])
    panel_b(ax_b, dec)
    panel_c(ax_c, units_all)
    panel_e(ax_d, units_all)
    panel_f(ax_e, units_all)

    fig.canvas.draw()
    shift_axes([ax_b], dy=-0.040)
    a_matrix_box = axes_a[1].get_position()
    d_box = ax_d.get_position()
    d_dx = max(0, min(0.030, a_matrix_box.x0 - d_box.x0))
    shift_axes([ax_d], dx=d_dx, dy=0.022, dw=-0.050)
    shift_axes([ax_e], dx=0.050, dy=0.022, dw=-0.025)

    handles = [
        Line2D([0], [0], marker=MARKERS[c], color="none", markerfacecolor="white", markeredgecolor=CITY_GREY, markersize=4, label=c)
        for c in CITY_ORDER
    ]
    handles.extend(
        [
            Line2D([0], [0], marker=EXTERNAL_MARKERS[c], color="none", markerfacecolor="white",
                   markeredgecolor=CITY_GREY, markersize=4, label=c)
            for c in ["Marseille", "Sydney"]
        ]
    )
    d_box = ax_d.get_position()
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(d_box.x0 + 0.015, d_box.y1 + 0.034),
        ncol=5,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.7,
    )
    fig.canvas.draw()
    add_panel_letters_figure1(fig, axes_a, ax_b, ax_c, ax_d, ax_e)

    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG, format="svg")
    fig.savefig(OUT_PNG, format="png", dpi=600)
    print(OUT_SVG)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
