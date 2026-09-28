"""p25_rebuild_models.py -- every model-derived number the manuscript quotes,
recomputed under the single specification used throughout the paper.

Single specification: every covariate is an indicator for "the first linked
article reported X", no missing-category indicators, jurisdiction (states with
fewer than 40 events pooled) and year fixed effects, NB2 dispersion fixed at
alpha = 5.579, jurisdiction-clustered bootstrap (2,000 replications for the
intensity model, 1,000 for the recurrence model as in p11).

Computes
  intensity   main block and secondary block (with bootstrap draws)
  sensitivity accepted-only subset, excluding 2016, distinct-outlet outcome,
              dropping the most-covered 0.1% of events
  archetypes  predicted E[Y | Y>=1] for case archetypes, standardized over the
              observed jurisdictions, years, and road classes
  recurrence  discrete-time logit with state and year effects; odds ratios,
              population-averaged daily hazard, and 30-day cumulative follow-up
              standardized by driver-flight x charges stratum
  multiplicity Bonferroni intervals for the 18 interval-based comparisons
              (10 intensity + 8 recurrence covariates), from bootstrap SEs

Writes data_build/rebuild.json and data_build/rebuild_draws.npz.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ztnb  # noqa: E402

ROOT = HERE.parent
DB = ROOT / "data_build"
ALPHA = 5.579314501383443
NBOOT = int(__import__("os").environ.get("NBOOT", 2000))
SEED = 2027
STATE_MIN = 40
MAIN = ["child", "teen", "elderly", "female", "victim_named",
        "hit_run_flag", "multi_fatality", "weekend"]
SEC = MAIN + ["dark", "charges_reported"]
HAZ_COVS = ["child", "teen", "elderly", "female", "victim_named",
            "hit_run_flag", "charges_reported", "multi_fatality"]
MAXDAY = 30
HAZ_NBOOT, HAZ_SEED = int(__import__("os").environ.get("HAZ_NBOOT", 1000)), 7

ARCHETYPES = [
    ("Baseline", {}),
    ("Child victim", {"child": 1}),
    ("Victim named", {"victim_named": 1}),
    ("Charges already reported", {"charges_reported": 1}),
    ("Driver fled", {"hit_run_flag": 1}),
    ("Multi-fatality", {"multi_fatality": 1}),
    ("Multi-fatality + fled + weekend", {"multi_fatality": 1, "hit_run_flag": 1, "weekend": 1}),
]


# ------------------------------------------------------------ intensity (as p15)
def collapse(df):
    vc = df["state"].value_counts()
    keep = set(vc[vc >= STATE_MIN].index)
    return df["state"].where(df["state"].isin(keep), "other")


def design(d, st, covs, road=False):
    cols, names = [np.ones(len(d))], ["const"]
    for c in covs:
        cols.append(d[c].to_numpy(float)); names.append(c)
    if road:
        for lev in sorted(d["road_class"].unique())[1:]:
            cols.append((d["road_class"] == lev).to_numpy(float)); names.append("rd_" + str(lev))
    for lev in sorted(st.unique())[1:]:
        cols.append((st == lev).to_numpy(float)); names.append("st_" + lev)
    for lev in sorted(d["year"].unique())[1:]:
        cols.append((d["year"] == lev).to_numpy(float)); names.append("yr_%s" % lev)
    return np.column_stack(cols), names


def fit(y, X, start=None):
    lna = np.log(ALPHA); k = X.shape[1]
    f = lambda b: ztnb.nll(np.r_[b, lna], y, X)
    g = lambda b: ztnb.grad(np.r_[b, lna], y, X)[:k]
    b0 = start if start is not None else np.r_[np.log(max(y.mean(), 1.01)), np.zeros(k - 1)]
    r = optimize.minimize(f, b0, jac=g, method="L-BFGS-B", options=dict(maxiter=400, ftol=1e-12))
    return r.x, bool(r.success)


def intensity(df, covs, road, yvar="n_articles", tag=""):
    t0 = time.time()
    st = collapse(df)
    y = df[yvar].to_numpy(float)
    X, names = design(df, st, covs, road)
    beta, ok = fit(y, X)
    rng = np.random.RandomState(SEED)
    sts = st.to_numpy(); uniq = np.unique(sts)
    idx = {s: np.flatnonzero(sts == s) for s in uniq}
    D, fails = [], 0
    for _ in range(NBOOT):
        rows = np.concatenate([idx[s] for s in rng.choice(uniq, size=len(uniq), replace=True)])
        bb, okb = fit(y[rows], X[rows], start=beta)
        if okb:
            D.append(bb)
        else:
            fails += 1
    D = np.array(D)
    print("  %-26s n=%5d  converged=%s  boot ok %d fail %d  (%.0f s)"
          % (tag, len(df), ok, len(D), fails, time.time() - t0), flush=True)
    return dict(beta=beta, names=names, X=X, draws=D, ok=ok, n=len(df), fails=fails)


def summarize(m, covs):
    out = {"n": m["n"], "converged": m["ok"], "nboot_ok": int(len(m["draws"])),
           "nboot_fail": m["fails"], "rr": {}, "ci": {}, "se_log": {}}
    for c in covs:
        j = m["names"].index(c)
        out["rr"][c] = float(np.exp(m["beta"][j]))
        out["ci"][c] = [float(np.percentile(np.exp(m["draws"][:, j]), 2.5)),
                        float(np.percentile(np.exp(m["draws"][:, j]), 97.5))]
        out["se_log"][c] = float(np.std(m["draws"][:, j], ddof=1))
    return out


def archetype_preds(m):
    names, X = m["names"], m["X"]
    flag_cols = [names.index(c) for c in SEC]

    def pred(b, delta):
        Xa = X.copy()
        Xa[:, flag_cols] = 0.0
        for c, v in delta.items():
            Xa[:, names.index(c)] = v
        return float(ztnb.cond_mean(np.exp(np.clip(Xa @ b, -30, 30)), ALPHA).mean())

    out = {}
    for lab, delta in ARCHETYPES:
        p = pred(m["beta"], delta)
        dr = np.array([pred(b, delta) for b in m["draws"]])
        out[lab] = {"cues": delta, "pred": p,
                    "ci": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]}
    return out


# ------------------------------------------------------------ recurrence (as p11)
def expand(df):
    corpus_end = df["last_pub"].max()
    fp = df["first_pub"]
    Ti = np.minimum(MAXDAY, (corpus_end - fp).dt.days).clip(lower=1).astype(int).values
    fu = df["followup_day"].values
    has_ev = (df["got_followup"].values == 1) & (np.nan_to_num(fu, nan=1e9) <= Ti)
    stop = np.where(has_ev, np.nan_to_num(fu, nan=0).astype(int), Ti)
    stop = np.maximum(stop, 1)
    ev_idx = np.repeat(np.arange(len(df)), stop)
    t = np.concatenate([np.arange(1, s + 1) for s in stop])
    event = np.zeros(len(t), dtype=int)
    end_pos = np.cumsum(stop) - 1
    event[end_pos[has_ev]] = 1
    base = df.iloc[ev_idx]
    per = np.where(t <= 7, "d" + pd.Series(t).astype(str).values,
                   np.where(t <= 14, "w2", np.where(t <= 21, "w3", "w4")))
    out = pd.DataFrame({"per": per, "st": base["st"].values, "yr": base["year"].values, "event": event})
    for c in HAZ_COVS:
        out[c] = base[c].astype(float).values
    out["n"] = 1
    agg = out.groupby(["per", "st", "yr"] + HAZ_COVS, as_index=False, observed=True).agg(
        ev=("event", "sum"), n=("n", "sum"))
    return agg


def haz_design(agg):
    cols, names = [], []
    for lev in sorted(agg["per"].unique()):
        cols.append((agg["per"] == lev).to_numpy(float)); names.append("per_" + str(lev))
    for c in HAZ_COVS:
        cols.append(agg[c].to_numpy(float)); names.append(c)
    for lev in sorted(agg["st"].unique())[1:]:
        cols.append((agg["st"] == lev).to_numpy(float)); names.append("st_" + str(lev))
    for lev in sorted(agg["yr"].unique())[1:]:
        cols.append((agg["yr"] == lev).to_numpy(float)); names.append("yr_%s" % lev)
    return np.column_stack(cols), names


def irls(X, ev, n, w=None, start=None, maxit=40, tol=1e-9, ridge=1e-8):
    nn = n if w is None else n * w
    p = np.divide(ev, np.maximum(n, 1e-12))
    k = X.shape[1]
    b = np.zeros(k) if start is None else start.copy()
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        v = np.clip(mu * (1 - mu), 1e-10, None)
        z = eta + (p - mu) / v
        W = nn * v
        A = X.T @ (X * W[:, None]) + ridge * np.eye(k)
        nb = np.linalg.solve(A, X.T @ (W * z))
        if not np.all(np.isfinite(nb)):
            return b, False
        if np.max(np.abs(nb - b)) < tol:
            return nb, True
        b = nb
    return b, True


def recurrence(df):
    t0 = time.time()
    df = df.copy()
    df["st"] = collapse(df)
    agg = expand(df)
    ev = agg["ev"].to_numpy(float); n = agg["n"].to_numpy(float)
    X, names = haz_design(agg)
    b, _ = irls(X, ev, n)
    rng = np.random.RandomState(HAZ_SEED)
    sts = agg["st"].to_numpy(); uniq, inv = np.unique(sts, return_inverse=True)
    nS = len(uniq); D = []
    for _ in range(HAZ_NBOOT):
        cnt = np.bincount(rng.randint(0, nS, nS), minlength=nS).astype(float)
        bb, ok = irls(X, ev, n, w=cnt[inv], start=b, maxit=12)
        if ok:
            D.append(bb)
    D = np.array(D)
    out = {"or": {}, "ci": {}, "se_log": {}, "nboot_ok": int(len(D))}
    for c in HAZ_COVS:
        j = names.index(c)
        out["or"][c] = float(np.exp(b[j]))
        out["ci"][c] = [float(np.percentile(np.exp(D[:, j]), 2.5)),
                        float(np.percentile(np.exp(D[:, j]), 97.5))]
        out["se_log"][c] = float(np.std(D[:, j], ddof=1))

    # per-event linear predictor pieces for standardized predictions
    j_st = {nm[3:]: i for i, nm in enumerate(names) if nm.startswith("st_")}
    j_yr = {nm[3:]: i for i, nm in enumerate(names) if nm.startswith("yr_")}
    fe = np.array([(b[j_st[s]] if s in j_st else 0.0) + (b[j_yr[str(y)]] if str(y) in j_yr else 0.0)
                   for s, y in zip(df["st"], df["year"])])
    per_b = {nm[4:]: b[i] for i, nm in enumerate(names) if nm.startswith("per_")}
    day_per = ["d%d" % t if t <= 7 else "w2" if t <= 14 else "w3" if t <= 21 else "w4"
               for t in range(1, MAXDAY + 1)]
    cov = {c: df[c].to_numpy(float) for c in HAZ_COVS}

    def curves(over):
        xb = fe + sum(b[names.index(c)] * (over.get(c, cov[c])) for c in HAZ_COVS)
        S = np.ones(len(df)); cum, haz = [], []
        for t in range(MAXDAY):
            h = 1 / (1 + np.exp(-(per_b[day_per[t]] + xb)))
            haz.append(float(h.mean()))
            S = S * (1 - h)
            cum.append(float((1 - S).mean()))
        return cum, haz

    strata = {"No hit-run, no charges": (0, 0), "Hit-run, no charges": (1, 0),
              "No hit-run, charges": (0, 1), "Hit-run + charges": (1, 1)}
    out["cum_by_stratum"] = {}
    for lab, (hr, ch) in strata.items():
        cum, _ = curves({"hit_run_flag": np.full(len(df), float(hr)),
                         "charges_reported": np.full(len(df), float(ch))})
        out["cum_by_stratum"][lab] = cum
    cum_obs, haz_obs = curves({})
    out["pop_hazard"] = haz_obs
    out["pop_cum30_model"] = cum_obs[-1]
    out["pop_cum"] = cum_obs
    out["observed_followup_rate"] = float(df["got_followup"].mean())
    print("  %-26s person-period cells %d  boot ok %d  (%.0f s)"
          % ("recurrence", len(agg), len(D), time.time() - t0), flush=True)
    return out, D


# ------------------------------------------------------------ run
def main():
    fa = pd.read_parquet(DB / "events_firstart.parquet")
    res, draws = {}, {}
    print("intensity models", flush=True)
    m_main = intensity(fa, MAIN, False, tag="main")
    res["main"] = summarize(m_main, MAIN); draws["main"] = m_main["draws"]
    m_sec = intensity(fa, SEC, True, tag="secondary")
    res["secondary"] = summarize(m_sec, SEC); draws["secondary"] = m_sec["draws"]
    res["archetypes"] = archetype_preds(m_sec)

    sens = {
        "accepted_only": (fa[fa["review_accepted"] == 1], "n_articles"),
        "exclude_2016": (fa[fa["year"] != 2016], "n_articles"),
        "distinct_outlets": (fa, "n_outlets"),
        "drop_top_0.1pct": (fa[fa["n_articles"] <= fa["n_articles"].quantile(0.999)], "n_articles"),
    }
    res["sensitivity"] = {}
    for k, (d, yv) in sens.items():
        m = intensity(d.reset_index(drop=True), MAIN, False, yvar=yv, tag=k)
        res["sensitivity"][k] = summarize(m, MAIN)

    print("recurrence model", flush=True)
    rec, Dh = recurrence(fa)
    res["recurrence"] = rec; draws["recurrence"] = Dh

    # Bonferroni over the 18 interval-based comparisons, from bootstrap SEs
    z = float(norm.ppf(1 - 0.05 / (2 * 18)))
    bon = {}
    for c in MAIN:
        r, s = res["main"]["rr"][c], res["main"]["se_log"][c]
        bon["intensity:" + c] = [r, float(np.exp(np.log(r) - z * s)), float(np.exp(np.log(r) + z * s))]
    for c in ["dark", "charges_reported"]:
        r, s = res["secondary"]["rr"][c], res["secondary"]["se_log"][c]
        bon["intensity:" + c] = [r, float(np.exp(np.log(r) - z * s)), float(np.exp(np.log(r) + z * s))]
    for c in HAZ_COVS:
        r, s = rec["or"][c], rec["se_log"][c]
        bon["recurrence:" + c] = [r, float(np.exp(np.log(r) - z * s)), float(np.exp(np.log(r) + z * s))]
    res["bonferroni"] = {"z": z, "comparisons": 18, "intervals": bon,
                         "survivors": [k for k, (_, lo, hi) in bon.items() if lo > 1 or hi < 1]}

    json.dump(res, open(DB / "rebuild.json", "w"), indent=1)
    np.savez_compressed(DB / "rebuild_draws.npz", **draws)
    print("\nwrote", DB / "rebuild.json")

    # console summary
    def row(tag, d):
        print("  %-18s " % tag + "  ".join("%s %.2f" % (c[:6], d["rr"][c]) for c in MAIN))
    row("main", res["main"])
    for k, v in res["sensitivity"].items():
        row(k, v)
    print("  secondary: dark %.2f %s  charges %.2f %s" % (
        res["secondary"]["rr"]["dark"], [round(x, 2) for x in res["secondary"]["ci"]["dark"]],
        res["secondary"]["rr"]["charges_reported"],
        [round(x, 2) for x in res["secondary"]["ci"]["charges_reported"]]))
    for lab, a in res["archetypes"].items():
        print("  archetype %-34s %.2f [%.2f, %.2f]" % (lab, a["pred"], a["ci"][0], a["ci"][1]))
    print("  recurrence ORs:", {c: round(v, 2) for c, v in rec["or"].items()})
    print("  30-day cumulative by stratum:", {k: round(v[-1], 4) for k, v in rec["cum_by_stratum"].items()})
    print("  population hazard d1 %.4f d2 %.4f d3 %.4f d15 %.5f d22 %.5f d30 %.5f" % tuple(
        rec["pop_hazard"][i] for i in (0, 1, 2, 14, 21, 29)))
    print("  model 30-day %.4f vs observed follow-up %.4f" % (rec["pop_cum30_model"], rec["observed_followup_rate"]))
    print("  Bonferroni survivors:", res["bonferroni"]["survivors"])


if __name__ == "__main__":
    main()
