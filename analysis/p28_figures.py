"""p28_figures.py -- Figures 3 and 4 under the single specification.

Figure 3 uses the drawing code of p03_ztnb.py for panels (a) and (b) unchanged
and redraws panel (c) for the archetypes computed in p25 (standardized over the
observed jurisdictions, years, and road classes, and labelled per event).
Figure 4 uses the drawing code of p04_hazard_gini.py with the recurrence
results of p27 (first-article covariates, state and year effects,
population-averaged hazard and standardized cumulative follow-up) and the
inequality statistics recomputed here; only axis ranges and labels that the new
values require are adjusted.

Reads data_build/rebuild.json and data_build/method_checks.json.
Writes figures/fig3.png, figures/fig4.png and copies both to the manuscript.
"""
from __future__ import annotations
import inspect, json, shutil, sys, textwrap
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import p03_ztnb as P3  # noqa: E402
import p04_hazard_gini as P4  # noqa: E402
from common import PED, HIGHLIGHT, SECONDARY, TERTIARY, NEUTRAL, INK  # noqa: E402

DB = C.DATA_BUILD
MS = C.ROOT / "outputs" / "figures"
reb = json.load(open(DB / "rebuild.json"))
chk = json.load(open(DB / "method_checks.json"))


# ------------------------------------------------------------ Figure 3
def rr_triplets(block):
    return {c: (block["rr"][c], block["ci"][c][0], block["ci"][c][1]) for c in block["rr"]}


main_rr = rr_triplets(reb["main"])
acc_rr = rr_triplets(reb["sensitivity"]["accepted_only"])

CUES = ("Child", "Named", "Charges", "Fled", "Weekend", "Multi")
CUE_KEY = {"Child": "child", "Named": "victim_named", "Charges": "charges_reported",
           "Fled": "hit_run_flag", "Weekend": "weekend", "Multi": "multi_fatality"}
CUE_COL = {"Child": PED, "Named": SECONDARY, "Charges": SECONDARY,
           "Fled": HIGHLIGHT, "Weekend": TERTIARY, "Multi": HIGHLIGHT}
ROW_LABEL = {
    "Baseline": "Baseline\n(no reported cue)",
    "Child victim": "Child victim",
    "Victim named": "Victim named",
    "Charges already reported": "Charges already\nreported",
    "Driver fled": "Driver fled",
    "Multi-fatality": "Multi-fatality",
    "Multi-fatality + fled + weekend": "Multi-fatality + fled\n+ weekend",
}
ROW_COLOR = {"Baseline": NEUTRAL, "Child victim": PED, "Victim named": SECONDARY,
             "Charges already reported": SECONDARY, "Driver fled": HIGHLIGHT,
             "Multi-fatality": HIGHLIGHT, "Multi-fatality + fled + weekend": HIGHLIGHT}


def archetype_panel(ax_cues, ax_eff, arche, corpus_mean):
    labs = list(arche)
    y = np.arange(len(labs))[::-1]
    for yi, lab in zip(y, labs):
        cues = arche[lab]["cues"]
        for xi, cue in enumerate(CUES):
            active = cues.get(CUE_KEY[cue], 0) == 1
            ax_cues.scatter(xi, yi, marker="s", s=28,
                            facecolor=CUE_COL[cue] if active else "#EDF2F5",
                            edgecolor="white" if active else "#D9E1E6", linewidth=0.45, zorder=2)
    ax_cues.set_xlim(-0.55, len(CUES) - 0.45)
    ax_cues.set_ylim(-0.5, len(labs) - 0.5)
    ax_cues.set_xticks(np.arange(len(CUES)))
    ax_cues.set_xticklabels(["Child", "Named", "Charges", "Fled", "Weekend", "Multi"],
                            fontsize=5.3, rotation=32, ha="right", rotation_mode="anchor")
    ax_cues.set_yticks(y)
    ax_cues.set_yticklabels([ROW_LABEL[l] for l in labs])
    ax_cues.set_ylabel("Case archetype")
    ax_cues.grid(False)
    ax_cues.tick_params(axis="both", length=0)
    ax_cues.tick_params(axis="x", pad=1)
    ax_cues.spines["bottom"].set_color("#BFC7CC")
    P3._panel_title(ax_cues, "c", "Model-Implied Coverage by Case Archetype")

    pts = [arche[l]["pred"] for l in labs]
    for yi, lab in zip(y, labs):
        p = arche[lab]["pred"]; lo, hi = arche[lab]["ci"]; col = ROW_COLOR[lab]
        ax_eff.hlines(yi, corpus_mean, p, color="#DCE9F0", linewidth=5.0, zorder=1)
        ax_eff.hlines(yi, lo, hi, color=col, linewidth=2.1, zorder=2)
        ax_eff.scatter(p, yi, s=42, color=col, edgecolor="white", linewidth=0.55, zorder=3)
        ax_eff.text(hi + 0.02, yi, f"{p:.2f}", color=col, va="center", ha="left",
                    fontsize=6.4, fontweight="bold")
    lo_all = min(arche[l]["ci"][0] for l in labs); hi_all = max(arche[l]["ci"][1] for l in labs)
    ax_eff.axvline(corpus_mean, color=INK, linestyle=":", linewidth=0.85, zorder=0)
    lift = pts[-1] - pts[0]
    bx = hi_all + 0.16
    ax_eff.annotate("", xy=(bx, y[-1]), xytext=(bx, y[0]),
                    arrowprops=dict(arrowstyle="<->", color=HIGHLIGHT, lw=0.8), zorder=2)
    ax_eff.text(bx + 0.015, (y[0] + y[-1]) / 2, f"+{lift:.2f}\n({lift / pts[0]:.0%})",
                fontsize=5.9, color=HIGHLIGHT, fontweight="bold", ha="left", va="center")
    ax_eff.set_xlim(lo_all - 0.08, bx + 0.16)
    ax_eff.set_xlabel(r"Predicted articles per event, $E[Y \mid Y \geq 1]$")
    ax_eff.grid(False)
    ax_eff.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax_eff.tick_params(axis="y", left=False, labelleft=False)
    ax_eff.spines["left"].set_visible(False)
    ax_eff.text(0.42, 0.97, "Filled squares = reported cue\nDotted rule = corpus mean; line = 95% CI",
                transform=ax_eff.transAxes, fontsize=5.8, color=NEUTRAL, ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.5))


def fig3():
    plt = C.use_style()
    fig = plt.figure(figsize=(7.15, 6.05), constrained_layout=True, facecolor="white")
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.0, 1.10],
                             wspace=0.10, hspace=0.10)
    axa = fig.add_subplot(outer[0, 0]); axb = fig.add_subplot(outer[0, 1])
    sc = outer[1, :].subgridspec(1, 2, width_ratios=[1.48, 4.25], wspace=0.04)
    axc = fig.add_subplot(sc[0, 0]); axe = fig.add_subplot(sc[0, 1], sharey=axc)
    P3._coverage_effect_field(axa, main_rr)
    # panel (b) drawing code from p03, with the axis range and label offsets
    # set for the first-article estimates, which span 0.5 to 2.9
    src = textwrap.dedent(inspect.getsource(P3._editorial_stability_map))
    for x, z in [
        ('lim = (0.58, 4.45)', 'lim = (0.50, 2.90)'),
        ('"multi_fatality": (10, 13, "left", "bottom")', '"multi_fatality": (14, 14, "left", "bottom")'),
        ('"hit_run_flag": (9, 9, "left", "bottom")', '"hit_run_flag": (-12, 14, "right", "bottom")'),
        ('"victim_named": (9, -13, "left", "top")', '"victim_named": (14, -6, "left", "top")'),
        ('"female": (-9, -14, "right", "top")', '"female": (-16, 12, "right", "bottom")'),
    ]:
        assert x in src, x
        src = src.replace(x, z)
    ns = dict(P3.__dict__)
    exec(src, ns)
    ns["_editorial_stability_map"](axb, main_rr, acc_rr)
    archetype_panel(axc, axe, reb["archetypes"], float(np.mean(
        pd.read_parquet(DB / "events_firstart.parquet")["n_articles"])))
    fig.align_ylabels([axa, axb])
    fig.savefig(C.FIGURES / "fig3.png", dpi=600, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ Figure 4
def fig4():
    rec = chk["recurrence"]
    curves = {k: np.array(v) for k, v in rec["cum_by_stratum"].items()}
    haz = {t + 1: v for t, v in enumerate(rec["pop_hazard"])}
    df = pd.read_parquet(DB / "events_firstart.parquet")
    y_all = df["n_articles"].values
    G = P4.gini(y_all)
    G_ci = P4.gini_ci_clustered(y_all, df["state"].values)
    t1, t5, t10 = P4.top_share(y_all, .01), P4.top_share(y_all, .05), P4.top_share(y_all, .10)
    t1_ci = C.pctile_ci(P4._cluster_draws(y_all, df["state"].values, lambda v: P4.top_share(v, .01)))
    gini_year = {int(yr): P4.gini(g["n_articles"].values) for yr, g in df.groupby("year") if len(g) > 30}

    ymax = max(c[-1] for c in curves.values())
    top = 0.05 * np.ceil((ymax + 0.02) / 0.05)
    sc = top / 0.09
    src = textwrap.dedent(inspect.getsource(P4.fig4))
    rep = [
        ('axa.set_xlim(0.5, 31.0); axa.set_ylim(0, 0.09)', 'axa.set_xlim(0.5, 31.0); axa.set_ylim(0, %r)' % top),
        ('axa.set_yticks([0, 0.025, 0.05, 0.075])', 'axa.set_yticks(%r)' % [round(x, 2) for x in np.arange(0, top + 1e-9, 0.05)]),
        ('axa.text(1.05, 0.084, "Day 1"', 'axa.text(1.05, %r, "Day 1"' % (0.084 * sc)),
        # the key sits in the open field below the lowest curve, right of day 6
        ('axa.text(2.0, 0.019, "30-DAY FOLLOW-UP"', 'axa.text(6.5, %r, "30-DAY FOLLOW-UP"' % (0.041 * top / 0.20)),
        ('("Hit-run + charges", "Both signals", 2.0, 0.0142)', '("Hit-run + charges", "Both signals", 6.5, %r)' % (0.031 * top / 0.20)),
        ('("Hit-run, no charges", "Hit-and-run", 16.1, 0.0142)', '("Hit-run, no charges", "Hit-and-run", 19.0, %r)' % (0.031 * top / 0.20)),
        ('("No hit-run, charges", "Charges", 2.0, 0.0059)', '("No hit-run, charges", "Charges", 6.5, %r)' % (0.014 * top / 0.20)),
        ('("No hit-run, no charges", "No cues", 16.1, 0.0059)', '("No hit-run, no charges", "No cues", 19.0, %r)' % (0.014 * top / 0.20)),
        ('axb.set_xlim(0.5, 30.5); axb.set_ylim(floor, 0.04)', 'axb.set_xlim(0.5, 30.5); axb.set_ylim(floor, 0.15)'),
        ('axb.set_yticks([1e-4, 1e-3, 1e-2])', 'axb.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])'),
        ('axb.set_yticklabels(["0.01%", "0.1%", "1%"])', 'axb.set_yticklabels(["0.01%", "0.1%", "1%", "10%"])'),
        ('axb.set_ylabel("Baseline follow-up hazard")', 'axb.set_ylabel("Daily follow-up hazard")'),
        # panel (c): each top-share label sits above-left of its own point, and
        # the annual-Gini inset moves right so its ticks clear the main y axis
        ('(1, "Top 1%", (12, 12))', '(1, "Top 1%", (-6, 6))'),
        ('(5, "Top 5%", (10, 10))', '(5, "Top 5%", (-6, 6))'),
        ('ha="left", va="bottom" if offset[1] > 0 else "top"',
         'ha="right" if offset[0] < 0 else "left", va="bottom" if offset[1] > 0 else "top"'),
        ('axin = axc.inset_axes([0.06, 0.60, 0.43, 0.25])', 'axin = axc.inset_axes([0.17, 0.60, 0.40, 0.25])'),
        ('"By week 3\\n<0.02% per day"','"By week 3\\n%.2f%% per day" % (100 * max(haz[t] for t in range(15, 22)))'),
    ]
    for a, b in rep:
        assert a in src, a
        src = src.replace(a, b)
    ns = dict(P4.__dict__)
    exec(src, ns)
    ns["fig4"](curves, haz, y_all, gini_year, G, G_ci, t1, t1_ci, t5, t10)
    return dict(G=G, G_ci=G_ci, t1=t1, t1_ci=t1_ci, t5=t5, t10=t10,
                gini_year=gini_year, haz=haz, cum30={k: float(v[-1]) for k, v in curves.items()})


if __name__ == "__main__":
    fig3()
    s = fig4()
    for f in ("fig3.png", "fig4.png"):
        shutil.copy(C.FIGURES / f, MS / f)
    print("Figure 4 values:", json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 5) for kk, vv in v.items()})
                                           for k, v in s.items()}, default=float))
    print("archetypes:", {k: round(v["pred"], 2) for k, v in reb["archetypes"].items()})
