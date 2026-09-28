"""p00_fig1_pipeline.py -- TRB-P Fig 1: method pipeline schematic (matplotlib).

Left->right: GDELT discovery -> scrape/LLM extraction -> canonical events ->
bridge-table outcome construction -> three parallel model boxes -> stakeholder
outputs. Drawn with the shared style kit / palette (same identity as all figures).
"""
from __future__ import annotations
import common as C
from common import PED, AV, HIGHLIGHT, SECONDARY, TERTIARY, NEUTRAL, INK


def box(ax, x, y, w, h, text, fc, ec=INK, tc=INK, fs=7, lw=0.9, bold=False):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                       linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=5)


def arrow(ax, x0, y0, x1, y1, color=NEUTRAL):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                shrinkA=1, shrinkB=1))


def main(n_events=12874, n_articles=18843):
    plt = C.use_style()
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def tint(hexc, a):  # light tint of a role color for fills
        from matplotlib.colors import to_rgb
        r, g, b = to_rgb(hexc)
        return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)

    ling = tint(NEUTRAL, 0.10)
    lblue = tint(PED, 0.14)

    # --- ingestion chain (top row) ---
    y0, h = 0.70, 0.20
    box(ax, 0.01, y0, 0.15, h, "GDELT\ndiscovery", ling, fs=6.5)
    box(ax, 0.19, y0, 0.16, h, "Scrape +\nLLM extraction", ling, fs=6.5)
    box(ax, 0.38, y0, 0.17, h, f"Canonical events\n(n={n_events:,})", lblue, fs=6.5, bold=True)
    box(ax, 0.58, y0, 0.19, h, f"Bridge-table outcome\nconstruction\n(Y, O, timing; {n_articles:,} articles)",
        lblue, fs=6.0)
    for x0, x1 in [(0.16, 0.19), (0.35, 0.38), (0.55, 0.58)]:
        arrow(ax, x0, y0 + h / 2, x1, y0 + h / 2)

    # --- three parallel model boxes (middle) ---
    ym, hm = 0.36, 0.22
    box(ax, 0.30, ym, 0.20, hm, "ZTNB intensity\n(RQ1: rate ratios)", tint(PED, 0.18), ec=PED, fs=6.3)
    box(ax, 0.52, ym, 0.20, hm, "Discrete-time\nfollow-up hazard\n(RQ2)", tint(HIGHLIGHT, 0.20), ec=HIGHLIGHT, fs=6.3)
    box(ax, 0.74, ym, 0.20, hm, "Lorenz–Gini\ninequality (RQ3)", tint(SECONDARY, 0.20), ec=SECONDARY, fs=6.3)
    # from outcome box down to the three models
    arrow(ax, 0.675, y0, 0.40, ym + hm)
    arrow(ax, 0.675, y0, 0.62, ym + hm)
    arrow(ax, 0.675, y0, 0.84, ym + hm)

    # --- stakeholder outputs (bottom) ---
    yb, hb = 0.04, 0.18
    box(ax, 0.14, yb, 0.33, hb, "Newsroom coverage-equity\naudit metrics", ling, fs=6.3)
    box(ax, 0.53, yb, 0.33, hb, "Surveillance bias profile\nfor news-derived crash data", ling, fs=6.3)
    arrow(ax, 0.40, ym, 0.30, yb + hb)
    arrow(ax, 0.62, ym, 0.62, yb + hb)
    arrow(ax, 0.84, ym, 0.70, yb + hb)

    fig.savefig(C.FIGURES / "fig1.pdf", bbox_inches="tight")
    fig.savefig(C.FIGURES / "fig1.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[fig1] wrote fig1.pdf")


if __name__ == "__main__":
    main()
