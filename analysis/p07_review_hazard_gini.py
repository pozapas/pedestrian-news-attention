"""p07_review_hazard_gini.py -- phase 2 of the review re-estimation (fast).

  #5  refit the follow-up model on predictors rebuilt from the first article
  #9  carry the SAME state and year effects the intensity model carries
  #10 report discrete-time logit estimates as ODDS ratios and add a
      complementary log-log fit whose exp(beta) is a proportional-hazard ratio
  #13 state-clustered Gini interval as primary
  #19 formal trend test on the annual Gini series
  #25 fractional-weight top shares

Person-periods are collapsed to unique covariate-by-period cells and fitted as
grouped binomial data, which is the identical likelihood at a fraction of the
cost. Appends to data_build/review_reest.json.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = ROOT / "data_build"
OUT = DB / "review_reest.json"

SEED = 7
NBOOT_HAZ = 1000
NBOOT_GINI = 2000
HORIZON = 30
COVS = ["child", "teen", "elderly", "age_missing", "female", "gender_missing",
        "victim_named", "hit_run_flag", "hit_run_missing", "multi_fatality",
        "weekend"]

res = json.load(open(OUT))


def prep(df):
    vc = df["state"].value_counts()
    keep = set(vc[vc >= 40].index)
    df = df.copy()
    df["st"] = df["state"].where(df["state"].isin(keep), "other")
    fu = df["followup_day"].to_numpy()
    last = np.where(np.isnan(fu), HORIZON,
                    np.minimum(np.maximum(fu, 1), HORIZON)).astype(int)
    rep = np.repeat(np.arange(len(df)), last)
    t = np.concatenate([np.arange(1, L + 1) for L in last])
    ev = ((~np.isnan(fu))[rep]) & (fu[rep] == t)
    per = np.where(t <= 7, t, np.where(t <= 14, 8, np.where(t <= 21, 9, 10)))
    base = df.iloc[rep]
    key = pd.DataFrame({"per": per, "st": base["st"].values,
                        "yr": base["year"].values})
    for c in COVS:
        key[c] = base[c].values
    key["ev"] = ev.astype(int)
    key["n"] = 1
    agg = key.groupby(["per", "st", "yr"] + COVS, as_index=False,
                      observed=True).agg(ev=("ev", "sum"), n=("n", "sum"))
    return agg, int(len(key))


def design(agg, fe=True):
    cols, names = [], []
    for lev in sorted(agg["per"].unique()):
        cols.append((agg["per"] == lev).to_numpy(float)); names.append("per_%s" % lev)
    for c in COVS:
        cols.append(agg[c].to_numpy(float)); names.append(c)
    if fe:
        for lev in sorted(agg["st"].unique())[1:]:
            cols.append((agg["st"] == lev).to_numpy(float)); names.append("st_" + lev)
        for lev in sorted(agg["yr"].unique())[1:]:
            cols.append((agg["yr"] == lev).to_numpy(float)); names.append("yr_%s" % lev)
    return np.column_stack(cols), names


def irls(X, ev, n, w=None, link="logit", start=None, maxit=40, tol=1e-9, ridge=1e-8):
    """Grouped-binomial IRLS. w scales the cell counts (cluster bootstrap)."""
    nn = n if w is None else n * w
    p_obs = np.divide(ev, np.maximum(n, 1e-12))
    k = X.shape[1]
    b = np.zeros(k) if start is None else start.copy()
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        if link == "logit":
            mu = 1.0 / (1.0 + np.exp(-eta))
            v = np.clip(mu * (1 - mu), 1e-10, None)
            z = eta + (p_obs - mu) / v
            W = nn * v
        else:
            e = np.exp(np.clip(eta, -30, 30))
            mu = np.clip(1.0 - np.exp(-e), 1e-12, 1 - 1e-12)
            d = np.clip(e * np.exp(-e), 1e-12, None)
            v = np.clip(mu * (1 - mu), 1e-12, None)
            z = eta + (p_obs - mu) / d
            W = nn * (d ** 2) / v
        A = X.T @ (X * W[:, None]) + ridge * np.eye(k)
        nb = np.linalg.solve(A, X.T @ (W * z))
        if not np.all(np.isfinite(nb)):
            return b, False
        if np.max(np.abs(nb - b)) < tol:
            return nb, True
        b = nb
    return b, True


def hazard_block(df, tag):
    agg, nrows = prep(df)
    ev = agg["ev"].to_numpy(float); n = agg["n"].to_numpy(float)
    Xf, nf = design(agg, fe=True)
    Xn, nn_ = design(agg, fe=False)
    b_fe, _ = irls(Xf, ev, n, link="logit")
    b_no, _ = irls(Xn, ev, n, link="logit")
    b_cll, _ = irls(Xf, ev, n, link="cloglog", start=b_fe * 0.5)

    rng = np.random.RandomState(SEED)
    sts = agg["st"].to_numpy(); uniq, inv = np.unique(sts, return_inverse=True)
    nS = len(uniq)
    draws, fails = [], 0
    for _ in range(NBOOT_HAZ):
        cnt = np.bincount(rng.randint(0, nS, nS), minlength=nS).astype(float)
        bb, ok = irls(Xf, ev, n, w=cnt[inv], link="logit", start=b_fe, maxit=12)
        draws.append(bb) if ok else None
        fails += (0 if ok else 1)
    D = np.array(draws)
    out = {"person_periods": nrows, "cells": int(len(agg)),
           "nboot_requested": NBOOT_HAZ, "nboot_ok": int(len(draws)),
           "nboot_fail": int(fails),
           "or_with_fe": {c: float(np.exp(b_fe[nf.index(c)])) for c in COVS},
           "or_no_fe": {c: float(np.exp(b_no[nn_.index(c)])) for c in COVS},
           "hr_cloglog_fe": {c: float(np.exp(b_cll[nf.index(c)])) for c in COVS},
           "or_ci_with_fe": {c: [float(np.percentile(np.exp(D[:, nf.index(c)]), 2.5)),
                                 float(np.percentile(np.exp(D[:, nf.index(c)]), 97.5))]
                             for c in COVS}}
    print("%s: cells=%d boot ok=%d fail=%d" % (tag, len(agg), len(draws), fails))
    return out


base = pd.read_parquet(DB / "events.parquet")
res["hazard_event_level"] = hazard_block(base, "hazard event-level")
fa_path = DB / "events_firstart.parquet"
if fa_path.exists():
    res["hazard_first_article"] = hazard_block(pd.read_parquet(fa_path),
                                               "hazard first-article")

# ================================================ inequality (#13, #19, #25)
yv = base["n_articles"].to_numpy(float)
vc = base["state"].value_counts(); keep = set(vc[vc >= 40].index)
stv = base["state"].where(base["state"].isin(keep), "other").to_numpy()


def gini(a):
    a = np.sort(np.asarray(a, float)); n = len(a)
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * a.sum()) - (n + 1) / n)


def top_frac(a, p):
    a = np.sort(np.asarray(a, float))[::-1]
    n = len(a); k = p * n; kk = int(np.floor(k))
    return float((a[:kk].sum() + (k - kk) * (a[kk] if kk < n else 0.0)) / a.sum())


def top_floor(a, p):
    a = np.sort(np.asarray(a, float))[::-1]
    return float(a[:int(np.floor(p * len(a)))].sum() / a.sum())


res["gini_point"] = gini(yv)
res["top_share_fractional"] = {str(p): top_frac(yv, p) for p in (0.01, 0.05, 0.10)}
res["top_share_floor"] = {str(p): top_floor(yv, p) for p in (0.01, 0.05, 0.10)}
res["top1_n_events_floor"] = int(np.floor(0.01 * len(yv)))
res["top1_pct_actual_floor"] = float(100 * np.floor(0.01 * len(yv)) / len(yv))

u2 = np.unique(stv); idx = {s: np.flatnonzero(stv == s) for s in u2}
rng2 = np.random.RandomState(99)
gs, t1s = [], []
for _ in range(NBOOT_GINI):
    rows = np.concatenate([idx[s] for s in rng2.choice(u2, size=len(u2), replace=True)])
    gs.append(gini(yv[rows])); t1s.append(top_frac(yv[rows], 0.01))
res["gini_ci_state_cluster"] = [float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))]
res["top1_ci_state_cluster"] = [float(np.percentile(t1s, 2.5)), float(np.percentile(t1s, 97.5))]
rng3 = np.random.RandomState(99)
ge = [gini(yv[rng3.randint(0, len(yv), len(yv))]) for _ in range(NBOOT_GINI)]
res["gini_ci_event_level"] = [float(np.percentile(ge, 2.5)), float(np.percentile(ge, 97.5))]

yrs = sorted(base["year"].unique())
gy = {int(t): gini(base.loc[base["year"] == t, "n_articles"].to_numpy(float)) for t in yrs}
res["gini_by_year"] = gy
xa = np.array(sorted(gy)); ya = np.array([gy[t] for t in xa])
res["gini_trend_slope_per_year"] = float(np.polyfit(xa, ya, 1)[0])
rng4 = np.random.RandomState(123); sl = []
for _ in range(500):
    rows = np.concatenate([idx[s] for s in rng4.choice(u2, size=len(u2), replace=True)])
    sub = base.iloc[rows]; g2 = {}
    for t in xa:
        v = sub.loc[sub["year"] == t, "n_articles"].to_numpy(float)
        if len(v) > 30:
            g2[t] = gini(v)
    if len(g2) >= 5:
        xx = np.array(sorted(g2)); sl.append(np.polyfit(xx, [g2[t] for t in xx], 1)[0])
res["gini_trend_slope_ci"] = [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]
res["gini_trend_nboot"] = len(sl)

json.dump(res, open(OUT, "w"), indent=1)
print("phase 2 written to", OUT)
