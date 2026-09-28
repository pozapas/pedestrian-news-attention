"""p03_ztnb.py -- TRB-P Step 3: zero-truncated NB2 coverage-intensity models.

Emits tables/table2.tex (RR forest with \\ciglyph, main + accepted-only blocks),
figures/fig3.pdf (effect-to-scenario coverage atlas), and results['ztnb'].

Modeling notes (see qa.md / advisor log):
- ZT-NB2 dispersion alpha runs to the logarithmic-series limit for this
  heavy-tailed corpus (profile likelihood monotone in alpha; both our scipy MLE
  and statsmodels hit the boundary; statsmodels diverges). alpha is NOT finitely
  identified from above. The mean-model rate ratios RR=exp(beta) ARE our RQ1
  deliverable and are invariant to alpha across [2.7, 20] (verified). We therefore
  FIX alpha at the interior MLE obtained on the tail-trimmed (drop-top-1%) sample
  for reproducibility of the beta fit, report RRs with cluster-bootstrap CIs, and
  report the LR test vs ZT-Poisson (>= 5000, p<1e-6) to justify NB over Poisson.
- Bootstrap resamples states (clusters) and refits beta ONLY at fixed alpha
  (β is alpha-invariant; letting alpha run per replicate would wreck prediction CIs).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2
import common as C
from common import PED, HIGHLIGHT, SECONDARY, TERTIARY, NEUTRAL, INK
import ztnb
import palette

warnings.filterwarnings("ignore")

COVS_MAIN = ["child", "teen", "elderly", "age_missing", "female", "gender_missing",
             "victim_named", "hit_run_flag", "hit_run_missing", "multi_fatality", "weekend"]
COVS_CONTEXT = ["dark", "dark_missing", "charges_reported", "charges_missing"]
ROAD_DUMMIES = ["road_intersection", "road_highway", "road_other", "road_missing"]  # ref=arterial
STATE_MIN = 40           # collapse states with < this many events into 'other'
REF_STATE = "CA"
REF_YEAR = 2019
NBOOT = 200

# pretty labels for Table 2 / Fig 3
LABELS = {
    "child": "Child ($\\le$12)", "teen": "Teen (13--19)", "elderly": "Age $\\ge$65",
    "female": "Female victim", "victim_named": "Victim named",
    "hit_run_flag": "Hit-and-run", "multi_fatality": "Multi-fatality",
    "weekend": "Weekend", "dark": "Dark conditions", "charges_reported": "Charges reported",
}
MAIN_ROWS = ["child", "teen", "elderly", "female", "victim_named", "hit_run_flag",
             "multi_fatality", "weekend"]
CONTEXT_ROWS = ["dark", "charges_reported"]


# ------------------------------------------------------------------ design
def collapse_states(df):
    vc = df["state"].value_counts()
    small = set(vc[vc < STATE_MIN].index)
    return df["state"].where(~df["state"].isin(small), "other")


def make_design(d: pd.DataFrame, covs, road=False) -> pd.DataFrame:
    X = pd.DataFrame({c: d[c].astype(float).values for c in covs}, index=d.index)
    X["intercept"] = 1.0
    if road:
        rc = d["road_class"]
        X["road_intersection"] = (rc == "intersection").astype(float).values
        X["road_highway"] = (rc == "highway").astype(float).values
        X["road_other"] = (rc == "other").astype(float).values
        X["road_missing"] = (rc == "missing").astype(float).values
        # reference = 'arterial'
    stcol = collapse_states(d)
    st = pd.get_dummies(stcol, prefix="st").astype(float)
    st = st.drop(columns=[f"st_{REF_STATE}"], errors="ignore")
    yr = pd.get_dummies(d["year"], prefix="yr").astype(float)
    yr = yr.drop(columns=[f"yr_{REF_YEAR}"], errors="ignore")
    return pd.concat([X, st.set_index(d.index), yr.set_index(d.index)], axis=1)


def fit_beta_fixed_alpha(y, X, alpha, start=None, maxiter=400):
    lna = np.log(alpha)
    k = X.shape[1]
    if start is None:
        start = ztnb.poisson_init(y, X)
    f = lambda b: ztnb.nll(np.concatenate([b, [lna]]), y, X)
    g = lambda b: ztnb.grad(np.concatenate([b, [lna]]), y, X)[:-1]
    r = optimize.minimize(f, start, jac=g, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-6})
    return dict(beta=r.x, nll=float(r.fun), converged=bool(r.success))


def determine_alpha(df):
    """Interior alpha MLE on the drop-top-1% (tail-trimmed) sample."""
    d = df[df.n_articles <= df.n_articles.quantile(0.99)]
    X = make_design(d, COVS_MAIN).values
    y = d.n_articles.values.astype(float)
    # joint bounded fit to find interior lnalpha
    b0 = ztnb.poisson_init(y, X)
    res = optimize.minimize(
        ztnb.nll, np.concatenate([b0, [1.0]]), args=(y, X), jac=ztnb.grad,
        method="L-BFGS-B", bounds=[(None, None)] * X.shape[1] + [(-2, 3.5)],
        options={"maxiter": 800})
    return float(np.exp(res.x[-1]))


# ------------------------------------------------------------------ bootstrap
def bootstrap(d, covs, road, alpha, outcome, full_beta_named, seed=2027, nboot=NBOOT):
    """Cluster (state) bootstrap, refitting beta only at fixed alpha. Prebuilds
    the full design once and indexes rows per replicate, dropping all-zero dummy
    columns (states/years absent from a replicate) before fitting."""
    rng = np.random.RandomState(seed)
    Xfull = make_design(d, covs, road)
    cols = list(Xfull.columns)
    Xvals = Xfull.values.astype(float)
    yvals = d[outcome].values.astype(float)
    states = d["state"].values
    uniq = np.unique(states)
    idx_by = {c: np.where(states == c)[0] for c in uniq}      # precompute once
    start_full = np.array([full_beta_named.get(c, 0.0) for c in cols])

    draws, conv = [], 0
    for _ in range(nboot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in drawn])
        Xb = Xvals[idx]
        yb = yvals[idx]
        nz = Xb.any(axis=0)                                    # keep non-all-zero cols
        r = fit_beta_fixed_alpha(yb, Xb[:, nz], alpha, start=start_full[nz])
        conv += r["converged"]
        draws.append(dict(zip([c for c, k in zip(cols, nz) if k], r["beta"])))
    return draws, conv / nboot


def rr_ci(draws, name):
    b = np.array([d[name] for d in draws if name in d])
    return C.pctile_ci(np.exp(b))


# ------------------------------------------------------------------ Table 2
def fmt_rr(rr, lo, hi):
    return f"{rr:.2f} ({lo:.2f}--{hi:.2f})"

def write_table2(main, acc, ctx_main, ctx_acc, lmin=0.5, lmax=3.0):
    """main/acc/ctx_* are dicts name -> (rr, lo, hi)."""
    def glyph(v, hollow=False):
        rr, lo, hi = v
        cmd = "\\ciglyphhollow" if hollow else "\\ciglyph"
        return f"{cmd}{{{lo:.3f}}}{{{rr:.3f}}}{{{hi:.3f}}}{{{lmin}}}{{{lmax}}}"
    L = []
    L.append("% Auto-generated by p03_ztnb.py -- do not edit by hand.")
    L.append("\\begin{tabular}{l r@{\\hskip 2pt}c r@{\\hskip 2pt}c}")
    L.append("\\toprule")
    L.append(" & \\multicolumn{2}{c}{Main model} & \\multicolumn{2}{c}{Accepted-only} \\\\")
    L.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    L.append("Coverage-intensity covariate & RR (95\\% CI) & forest & RR (95\\% CI) & forest \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{5}{l}{\\emph{Near-universal covariates (main model)}}\\\\")
    for k in MAIN_ROWS:
        m, a = main[k], acc[k]
        L.append(f"\\quad {LABELS[k]} & {fmt_rr(*m)} & {glyph(m)} & "
                 f"{fmt_rr(*a)} & {glyph(a, hollow=True)} \\\\")
    L.append("\\addlinespace[2pt]")
    L.append("\\multicolumn{5}{l}{\\emph{Context covariates (secondary model, reported-attribute)}}\\\\")
    for k in CONTEXT_ROWS:
        m, a = ctx_main[k], ctx_acc[k]
        L.append(f"\\quad {LABELS[k]} & {fmt_rr(*m)} & {glyph(m)} & "
                 f"{fmt_rr(*a)} & {glyph(a, hollow=True)} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    (C.TABLES / "table2.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[table2] wrote table2.tex")


# ------------------------------------------------------------------ Fig 3
ARCHETYPES = [
    ("Baseline adult\n(day, no flags)", {}),
    ("Child, crosswalk\n(daytime)", {"child": 1}),
    ("Adult, mid-block,\nafter dark", {"dark": 1}),
    ("Elderly,\nhit-and-run", {"elderly": 1, "hit_run_flag": 1}),
    ("Named victim,\ncharges filed", {"victim_named": 1, "charges_reported": 1}),
    ("Child + hit-and-run\n+ named", {"child": 1, "hit_run_flag": 1, "victim_named": 1}),
]

def predict_archetype(beta_named, delta, alpha):
    eta = beta_named.get("intercept", 0.0) + sum(beta_named.get(k, 0.0) * v for k, v in delta.items())
    mu = np.exp(np.clip(eta, -30, 30))
    return float(ztnb.cond_mean(mu, alpha))

EFFECT_ORDER = ["multi_fatality", "hit_run_flag", "victim_named", "weekend",
                "child", "elderly", "teen", "female"]
EFFECT_LABELS = {
    "multi_fatality": "Multi-fatality\ncrash", "hit_run_flag": "Hit-and-run",
    "victim_named": "Victim named", "weekend": "Weekend", "child": "Child (≤12)",
    "elderly": "Older adult (≥65)", "teen": "Teen (13–19)", "female": "Female victim",
}


def _effect_color(name):
    """Fixed semantic palette: attention amplifiers, baseline signals, and disparity."""
    if name == "multi_fatality":
        return HIGHLIGHT
    if name == "female":
        return SECONDARY
    if name in {"hit_run_flag", "victim_named"}:
        return PED
    return NEUTRAL


def _panel_title(ax, letter: str, title: str):
    """Place every Figure 3 heading on the same left-anchored title baseline."""
    heading = ax.set_title(f"({letter})  {title}", loc="left", pad=10)
    heading.set_x(0)
    heading.set_ha("left")
    return heading


def _coverage_effect_field(ax, main_rr):
    """Panel A: compact, inferential view of the primary model's effect structure."""
    y = np.arange(len(EFFECT_ORDER))[::-1]
    for yi, name in zip(y, EFFECT_ORDER):
        rr = main_rr[name]
        color = _effect_color(name)
        significant = rr[1] > 1 or rr[2] < 1
        ax.hlines(yi, rr[1], rr[2], color=color, linewidth=2.1, zorder=2)
        ax.scatter(rr[0], yi, s=34, color=color if significant else "white",
                   edgecolor=color, linewidth=1.15, zorder=3)
        ax.text(4.37, yi, f"{rr[0]:.2f}", ha="right", va="center", fontsize=6.5,
                color=color if significant else NEUTRAL, fontweight="bold" if significant else "normal")

    ax.axvline(1, color=INK, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.70, 4.5)
    ax.set_xticks([0.75, 1, 1.5, 2, 3, 4])
    ax.set_xticklabels(["0.75", "1", "1.5", "2", "3", "4"])
    ax.set_yticks(y)
    ax.set_yticklabels([EFFECT_LABELS[k] for k in EFFECT_ORDER])
    ax.set_xlabel("Rate ratio (log scale)")
    ax.set_ylabel("Covariate")
    ax.grid(False)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    _panel_title(ax, "a", "Coverage-Intensity Associations")


def _editorial_stability_map(ax, main_rr, acc_rr):
    """Panel B: show whether full-corpus signals persist after editorial selection."""
    lim = (0.58, 4.45)
    ax.fill_between(lim, lim[0], lim[1], color="#F6F8F9", zorder=0)
    ax.plot(lim, lim, color=NEUTRAL, linewidth=0.85, linestyle="--", zorder=1)
    ax.axvline(1, color="#BFC7CC", linewidth=0.75, zorder=1)
    ax.axhline(1, color="#BFC7CC", linewidth=0.75, zorder=1)

    focus_offsets = {
        # Point offsets and opaque callout cards deliberately keep every label
        # clear of the diagonal, point estimate, and confidence intervals.
        "multi_fatality": (10, 13, "left", "bottom"),
        "hit_run_flag": (9, 9, "left", "bottom"),
        "victim_named": (9, -13, "left", "top"),
        "female": (-9, -14, "right", "top"),
    }
    focus_labels = {
        "multi_fatality": "Multi-fatality", "hit_run_flag": "Hit-and-run",
        "victim_named": "Victim named", "female": "Female victim",
    }
    for name in EFFECT_ORDER:
        full, acc = main_rr[name], acc_rr[name]
        color = _effect_color(name)
        ax.errorbar(full[0], acc[0],
                    xerr=[[full[0] - full[1]], [full[2] - full[0]]],
                    yerr=[[acc[0] - acc[1]], [acc[2] - acc[0]]],
                    fmt="none", ecolor=color, elinewidth=0.9, capsize=0, alpha=0.55, zorder=2)
        ax.scatter(full[0], acc[0], s=33 if name in focus_labels else 23,
                   color=color, edgecolor="white", linewidth=0.45, zorder=3)
        if name in focus_offsets:
            dx, dy, ha, va = focus_offsets[name]
            ax.annotate(focus_labels[name], (full[0], acc[0]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=5.8, color=INK, ha=ha, va=va, zorder=5,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.96, pad=1.0),
                        arrowprops=dict(arrowstyle="-", color="#BFC7CC", lw=0.55,
                                        shrinkA=1.5, shrinkB=2.0))

    ax.set_xlim(*lim); ax.set_ylim(*lim)
    # Equal aspect keeps the identity diagonal at 45 degrees. The outer
    # gridspec gives this cell a near-square footprint (width_ratios below), so
    # the box-adjusted square fills its cell instead of being shrunk and
    # centred; panel (b) therefore matches panel (a) in height and both the
    # titles and the x-axis labels sit on shared baselines. Using
    # adjustable="datalim" instead would expand the limits past `lim`, leaving
    # the shaded field and reference lines short of the axes edge.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Full-corpus rate ratio")
    ax.set_ylabel("Accepted-only rate ratio")
    ax.grid(False)
    ax.text(0.04, 0.96, "Dashed diagonal = identical association", transform=ax.transAxes,
            fontsize=6.0, color=NEUTRAL, va="top")
    _panel_title(ax, "b", "Editorial-Selection Stability")


CUE_COLUMNS = ("Child", "Dark", "Older", "Hit-run", "Named", "Charges")
CUE_COLORS = {
    "Child": PED, "Dark": TERTIARY, "Older": NEUTRAL,
    "Hit-run": HIGHLIGHT, "Named": SECONDARY, "Charges": SECONDARY,
}
ARCHETYPE_CUES = (
    set(), {"Child"}, {"Dark"}, {"Older", "Hit-run"},
    {"Named", "Charges"}, {"Child", "Hit-run", "Named"},
)


def _archetype_attention_ladder(ax_cues, ax_effect, sec_beta, sec_draws, alpha, calib):
    """Panel C: link each case recipe to its model-implied attention interval."""
    labels, points, los, his = [], [], [], []
    for lab, delta in ARCHETYPES:
        pt = predict_archetype(sec_beta, delta, alpha)
        draws = [predict_archetype(d, delta, alpha) for d in sec_draws]
        lo, hi = C.pctile_ci(draws)
        labels.append(lab.replace("\n", " "))
        points.append(pt); los.append(lo); his.append(hi)

    y = np.arange(len(labels))[::-1]
    colors = [NEUTRAL, PED, TERTIARY, HIGHLIGHT, SECONDARY, HIGHLIGHT]
    population_mean = calib["sample_mean_pred"]

    # Left-side recipe matrix: filled cells make the ingredient structure of
    # every model prediction visible before the reader reaches its interval.
    for yi, active_cues in zip(y, ARCHETYPE_CUES):
        for xi, cue in enumerate(CUE_COLUMNS):
            active = cue in active_cues
            ax_cues.scatter(xi, yi, marker="s", s=28,
                            facecolor=CUE_COLORS[cue] if active else "#EDF2F5",
                            edgecolor="white" if active else "#D9E1E6",
                            linewidth=0.45, zorder=2)
    ax_cues.set_xlim(-0.55, len(CUE_COLUMNS) - 0.45)
    ax_cues.set_ylim(-0.5, len(labels) - 0.5)
    ax_cues.set_xticks(np.arange(len(CUE_COLUMNS)))
    ax_cues.set_xticklabels(["Child", "Dark", "Older", "Hit-run", "Named", "Charges"],
                             fontsize=5.3, rotation=32, ha="right", rotation_mode="anchor")
    ax_cues.set_yticks(y)
    ax_cues.set_yticklabels([
        "Baseline adult\n(day, no flags)", "Child, crosswalk\n(daytime)",
        "Adult, mid-block\n(after dark)", "Older adult\n+ hit-and-run",
        "Named victim\n+ charges filed", "Child + hit-and-run\n+ named",
    ])
    ax_cues.set_ylabel("Case archetype")
    ax_cues.grid(False)
    ax_cues.tick_params(axis="both", length=0)
    ax_cues.tick_params(axis="x", pad=1)
    ax_cues.spines["bottom"].set_color("#BFC7CC")
    _panel_title(ax_cues, "c", "Model-Implied Coverage by Case Archetype")

    for yi, p, lo, hi, color in zip(y, points, los, his, colors):
        # The pale rail makes the distance from the observed corpus average visible,
        # while the saturated segment carries the model-implied interval.
        ax_effect.hlines(yi, population_mean, p, color="#DCE9F0", linewidth=5.0, zorder=1)
        ax_effect.hlines(yi, lo, hi, color=color, linewidth=2.1, zorder=2)
        ax_effect.scatter(p, yi, s=42, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax_effect.text(hi + 0.025, yi, f"{p:.2f}", color=color, va="center", ha="left",
                fontsize=6.4, fontweight="bold")

    # The bracket isolates the contrast between the reference and the stacked
    # cue profile without implying a causal sequence between the other rows.
    lift = points[-1] - points[0]
    bracket_x = 2.72
    ax_effect.axvline(population_mean, color=INK, linestyle=":", linewidth=0.85, zorder=0)
    ax_effect.annotate("", xy=(bracket_x, y[-1]), xytext=(bracket_x, y[0]),
                       arrowprops=dict(arrowstyle="<->", color=HIGHLIGHT, lw=0.8), zorder=2)
    ax_effect.text(bracket_x + 0.015, (y[0] + y[-1]) / 2,
                   f"+{lift:.2f}\n({lift / points[0]:.0%})", fontsize=5.9,
                   color=HIGHLIGHT, fontweight="bold", ha="left", va="center")
    ax_effect.set_xlim(1.30, 2.84)
    ax_effect.set_xlabel(r"Predicted articles per death, $E[Y \mid Y \geq 1]$")
    ax_effect.grid(False)
    ax_effect.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax_effect.tick_params(axis="y", left=False, labelleft=False)
    ax_effect.spines["left"].set_visible(False)
    ax_effect.text(0.02, 0.96,
                   "Filled squares = active model cue\nDotted rule = corpus mean; line = 95% CI",
                   transform=ax_effect.transAxes, fontsize=5.8, color=NEUTRAL, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.5))
    return {lab.replace(chr(10), " "): dict(pred=p, lo=lo, hi=hi)
            for lab, p, lo, hi in zip([a[0] for a in ARCHETYPES], points, los, his)}


def fig3(main_rr, acc_rr, sec_beta, sec_draws, alpha, calib):
    """Build an effect-to-scenario atlas for the coverage-intensity model."""
    plt = C.use_style()
    fig = plt.figure(figsize=(7.15, 6.05), constrained_layout=True, facecolor="white")
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92],
                             width_ratios=[1.0, 1.10], wspace=0.10, hspace=0.10)
    axa = fig.add_subplot(outer[0, 0])
    axb = fig.add_subplot(outer[0, 1])
    scenario = outer[1, :].subgridspec(1, 2, width_ratios=[1.48, 4.25], wspace=0.04)
    axc_cues = fig.add_subplot(scenario[0, 0])
    axc_effect = fig.add_subplot(scenario[0, 1], sharey=axc_cues)

    _coverage_effect_field(axa, main_rr)
    _editorial_stability_map(axb, main_rr, acc_rr)
    archetypes = _archetype_attention_ladder(axc_cues, axc_effect, sec_beta, sec_draws, alpha, calib)
    fig.align_ylabels([axa, axb])
    fig.savefig(C.FIGURES / "fig3.pdf", facecolor="white")
    fig.savefig(C.FIGURES / "fig3.png", dpi=600, facecolor="white")
    plt.close(fig)
    print("[fig3] wrote fig3.pdf")
    return archetypes


# ------------------------------------------------------------------ main
def main():
    print("[ztnb] unit test:", "PASS" if ztnb.unit_test(verbose=False) else "FAIL")
    df = C.load_events()
    y = df.n_articles.values.astype(float)

    alpha = determine_alpha(df)
    print(f"[alpha] fixed at interior MLE (drop-top-1% sample) = {alpha:.3f}")

    # ---- full fits ----
    Xm = make_design(df, COVS_MAIN)
    main_fit = fit_beta_fixed_alpha(y, Xm.values, alpha)
    bm = dict(zip(Xm.columns, main_fit["beta"]))

    Xs = make_design(df, COVS_MAIN + COVS_CONTEXT, road=True)
    sec_fit = fit_beta_fixed_alpha(y, Xs.values, alpha)
    bs = dict(zip(Xs.columns, sec_fit["beta"]))

    # separation check on covariates
    big = {k: v for k, v in {**bm, **bs}.items()
           if abs(v) > 5 and not k.startswith(("st_", "yr_", "intercept"))}
    fe_big = {k: v for k, v in {**bm, **bs}.items() if abs(v) > 5 and k.startswith(("st_", "yr_"))}
    print(f"[sep check] |beta|>5 covariates: {big if big else 'none'}; "
          f"FE with |beta|>5: {len(fe_big)}")

    # ---- LR test vs ZT-Poisson ----
    zp = ztnb.fit_ztp(y, Xm.values)
    # polish ZTP from its own solution (convex mean model)
    zp2 = ztnb.fit_ztp(y, Xm.values, start=zp["beta"])
    zp_nll = min(zp["nll"], zp2["nll"])
    LR = 2 * (zp_nll - main_fit["nll"])
    LR_p = float(chi2.sf(LR, 1))
    print(f"[LR] ZTNB vs ZT-Poisson = {LR:.1f}, p = {LR_p:.2e}")

    # ---- calibration (sample-averaged predicted cond mean) ----
    mu_all = np.exp(np.clip(Xm.values @ main_fit["beta"], -30, 30))
    pred_mean = float(ztnb.cond_mean(mu_all, alpha).mean())
    calib = dict(sample_mean_pred=pred_mean, observed_mean=float(y.mean()),
                 observed_median=float(np.median(y)))
    print(f"[calibration] mean predicted E[Y|Y>=1] = {pred_mean:.3f} "
          f"(observed mean {y.mean():.3f}, median {np.median(y):.0f})")

    # ---- bootstraps (beta only, fixed alpha) ----
    print(f"[bootstrap] main model ({NBOOT} reps)...")
    dm_draws, cm = bootstrap(df, COVS_MAIN, False, alpha, "n_articles", bm, seed=11)
    print(f"    converged {cm:.0%}")
    print(f"[bootstrap] accepted-only ({NBOOT} reps)...")
    dacc = df[df.review_accepted == 1]
    acc_fit = fit_beta_fixed_alpha(dacc.n_articles.values.astype(float),
                                   make_design(dacc, COVS_MAIN).values, alpha)
    bacc = dict(zip(make_design(dacc, COVS_MAIN).columns, acc_fit["beta"]))
    dacc_draws, ca = bootstrap(dacc, COVS_MAIN, False, alpha, "n_articles", bacc, seed=22)
    print(f"    converged {ca:.0%}")
    print(f"[bootstrap] secondary model ({NBOOT} reps)...")
    ds_draws, cs = bootstrap(df, COVS_MAIN + COVS_CONTEXT, True, alpha, "n_articles", bs, seed=33)
    print(f"    converged {cs:.0%}")
    # accepted-only secondary for context block
    sacc_fit = fit_beta_fixed_alpha(dacc.n_articles.values.astype(float),
                                    make_design(dacc, COVS_MAIN + COVS_CONTEXT, True).values, alpha)
    bsacc = dict(zip(make_design(dacc, COVS_MAIN + COVS_CONTEXT, True).columns, sacc_fit["beta"]))
    dsacc_draws, csa = bootstrap(dacc, COVS_MAIN + COVS_CONTEXT, True, alpha, "n_articles", bsacc, seed=44)

    # ---- assemble RRs ----
    def rr_row(beta_named, draws, k):
        return (float(np.exp(beta_named[k])), *rr_ci(draws, k))
    main_rr = {k: rr_row(bm, dm_draws, k) for k in MAIN_ROWS}
    acc_rr = {k: rr_row(bacc, dacc_draws, k) for k in MAIN_ROWS}
    ctx_main = {k: rr_row(bs, ds_draws, k) for k in CONTEXT_ROWS}
    ctx_acc = {k: rr_row(bsacc, dsacc_draws, k) for k in CONTEXT_ROWS}
    write_table2(main_rr, acc_rr, ctx_main, ctx_acc)

    # ---- Fig 3 ----
    arche = fig3(main_rr, acc_rr, bs, ds_draws, alpha, calib)

    # ---- sensitivities (point RRs) ----
    def sens_rr(d, outcome, covs=COVS_MAIN, road=False):
        X = make_design(d, covs, road)
        f = fit_beta_fixed_alpha(d[outcome].values.astype(float), X.values, alpha)
        bn = dict(zip(X.columns, f["beta"]))
        return {k: float(np.exp(bn[k])) for k in MAIN_ROWS}
    sens = {
        "exclude_2016": sens_rr(df[df.year != 2016], "n_articles"),
        "outcome_outlets": sens_rr(df, "n_outlets"),
        "drop_top_0.1pct": sens_rr(df[df.n_articles <= df.n_articles.quantile(0.999)], "n_articles"),
        "accepted_only": {k: acc_rr[k][0] for k in MAIN_ROWS},
    }

    # ---- results.json ----
    payload = {
        "alpha_fixed": alpha,
        "alpha_note": ("dispersion runs to the logarithmic-series limit; not finitely "
                       "identified from above; RRs invariant to alpha in [2.7,20]; "
                       "alpha fixed at interior MLE of drop-top-1% sample for reproducibility"),
        "lr_vs_ztpoisson": {"stat": float(LR), "p": LR_p, "df": 1,
                            "note": "boundary test; overwhelming overdispersion"},
        "n": int(len(df)),
        "nboot": NBOOT,
        "boot_convergence": {"main": cm, "accepted": ca, "secondary": cs},
        "sep_check": {"cov_beta_gt5": big, "n_fe_beta_gt5": len(fe_big)},
        "calibration": calib,
        "rr_main": {k: {"rr": v[0], "lo": v[1], "hi": v[2]} for k, v in main_rr.items()},
        "rr_accepted": {k: {"rr": v[0], "lo": v[1], "hi": v[2]} for k, v in acc_rr.items()},
        "rr_context": {k: {"rr": v[0], "lo": v[1], "hi": v[2]} for k, v in ctx_main.items()},
        "rr_context_accepted": {k: {"rr": v[0], "lo": v[1], "hi": v[2]} for k, v in ctx_acc.items()},
        "archetypes": arche,
        "sensitivities": sens,
    }
    C.update_results("ztnb", payload)
    print("[results] ztnb section written")


if __name__ == "__main__":
    main()
