from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

FIG_PNG = OUT / "Figure2_statistical_compression_mechanism_v1.png"
FIG_SVG = OUT / "Figure2_statistical_compression_mechanism_v1.svg"

cities = ["Tokyo", "Amsterdam", "London", "Marseille", "Sydney"]
base = np.array([0.180, 0.547, 0.273, 0.457, 0.312])
control = np.array([-0.066, 0.073, -0.041, -0.217, -0.091])
within = np.array([-0.042, 0.161, 0.007, -0.126, -0.040])

palette = {
    "teal": "#0E7180",
    "teal_light": "#86BFC1",
    "gold": "#C9902E",
    "gold_light": "#E6C466",
    "grey": "#73808A",
    "light": "#F4F2EB",
    "grid": "#E5E9EA",
}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8A969C")
    ax.spines["bottom"].set_color("#8A969C")
    ax.tick_params(labelsize=7, colors="#27323A", width=0.6)
    ax.grid(True, axis="x", color=palette["grid"], lw=0.6, zorder=0)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.2, 4.9), dpi=300)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1],
        width_ratios=[1.15, 1],
        hspace=0.42,
        wspace=0.35,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(cities))
    for yi, b, c in zip(y, base, control):
        ax1.plot([c, b], [yi, yi], color="#BEC8CB", lw=2.2, zorder=1)
    ax1.scatter(
        base,
        y,
        s=38,
        color=palette["gold"],
        edgecolor="white",
        lw=0.7,
        label="Before electricity control",
        zorder=3,
    )
    ax1.scatter(
        control,
        y,
        s=38,
        color=palette["teal"],
        edgecolor="white",
        lw=0.7,
        label="After electricity control",
        zorder=3,
    )
    ax1.axvline(0, color="#75858B", lw=0.8, ls="--")
    ax1.set_yticks(y)
    ax1.set_yticklabels(cities)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.3, 0.62)
    ax1.set_xlabel("SVI coefficient for proxy error")
    style_axis(ax1)
    ax1.grid(False, axis="y")
    ax1.legend(frameon=False, fontsize=6.5, loc="lower right")

    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.34
    ax2.barh(y + width / 2, control, height=width, color=palette["teal_light"], label="Controlled")
    ax2.barh(y - width / 2, within, height=width, color=palette["gold_light"], label="Within-electricity strata")
    ax2.axvline(0, color="#75858B", lw=0.8, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels(cities)
    ax2.invert_yaxis()
    ax2.set_xlim(-0.26, 0.22)
    ax2.set_xlabel("Residual SVI coefficient")
    style_axis(ax2)
    ax2.grid(False, axis="y")
    ax2.legend(frameon=False, fontsize=6.5, loc="lower right")

    ax3 = fig.add_subplot(gs[1, :])
    x = np.linspace(0, 100, 160)
    observed = x / 100
    predicted = 0.18 + 0.68 * observed
    ax3.plot(x, observed, color="#27323A", lw=1.2, label="Observed electricity")
    ax3.plot(x, predicted, color=palette["teal"], lw=1.8, label="Proxy prediction")
    ax3.fill_between(
        x,
        observed,
        predicted,
        where=predicted >= observed,
        color=palette["gold"],
        alpha=0.22,
        label="Positive proxy error",
    )
    ax3.fill_between(x, observed, predicted, where=predicted < observed, color=palette["teal"], alpha=0.14)
    ax3.axvspan(0, 25, color=palette["light"], zorder=-1)
    ax3.text(4, 0.86, "low-electricity tail", fontsize=7, color=palette["grey"])
    ax3.annotate(
        "regression toward the mean\nraises low-use predictions",
        xy=(13, 0.28),
        xytext=(32, 0.52),
        arrowprops=dict(arrowstyle="-|>", color=palette["gold"], lw=0.8),
        fontsize=7,
        color="#7A5419",
    )
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 1.03)
    ax3.set_xlabel("Observed-electricity percentile")
    ax3.set_ylabel("Relative electricity")
    style_axis(ax3)
    ax3.grid(True, axis="y", color=palette["grid"], lw=0.6)
    ax3.legend(frameon=False, fontsize=6.5, ncol=3, loc="lower right")

    for label, ax in zip(["a", "b", "c"], [ax1, ax2, ax3]):
        ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=8, weight="bold", va="top", ha="left")

    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_SVG, bbox_inches="tight", facecolor="white")
    print(FIG_PNG)
    print(FIG_SVG)


if __name__ == "__main__":
    main()
