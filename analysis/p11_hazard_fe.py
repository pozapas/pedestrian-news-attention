"""p11_hazard_fe.py -- Review #9, #10, #5 (recurrence model).

Replicates p04_hazard_gini.py's person-period construction EXACTLY (same
HAZ_COVS, same administrative censoring at min(30, corpus_end - first_pub),
same baseline bins) and then adds the state and year effects the intensity
model carries. Anything that differs between the published estimates and these
is therefore attributable to the added adjustment, not to a changed model.

Also refits on first-article predictors (#5) and reports complementary log-log
estimates (#10).

Writes data_build/hazard_fe.json.
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

HAZ_COVS = ["child", "teen", "elderly", "female", "victim_named",
            "hit_run_flag", "charges_reported", "multi_fatality"]
MAXDAY = 30
NBOOT = 1000
SEED = 7


def expand(df):
    """Identical construction to p04_hazard_gini.expand_person_period."""
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
    out = pd.DataFrame({"per": per, "st": base["st"].values,
                        "yr": base["year"].values, "event": event})
    for c in HAZ_COVS:
        out[c] = base[c].astype(float).values
    out["n"] = 1
    agg = out.groupby(["per", "st", "yr"] + HAZ_COVS, as_index=False,
                      observed=True).agg(ev=("event", "sum"), n=("n", "sum"))
    return agg, int(len(out))


def design(agg, fe):
    cols, names = [], []
    for lev in sorted(agg["per"].unique()):
        cols.append((agg["per"] == lev).to_numpy(float)); names.append("per_" + str(lev))
    for c in HAZ_COVS:
        cols.append(agg[c].to_numpy(float)); names.append(c)
    if fe:
        for lev in sorted(agg["st"].unique())[1:]:
            cols.append((agg["st"] == lev).to_numpy(float)); names.append("st_" + str(lev))
        for lev in sorted(agg["yr"].unique())[1:]:
            cols.append((agg["yr"] == lev).to_numpy(float)); names.append("yr_%s" % lev)
    return np.column_stack(cols), names


def irls(X, ev, n, w=None, link="logit", start=None, maxit=40, tol=1e-9, ridge=1e-8):
    nn = n if w is None else n * w
    p = np.divide(ev, np.maximum(n, 1e-12))
    k = X.shape[1]
    b = np.zeros(k) if start is None else start.copy()
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        if link == "logit":
            mu = 1.0 / (1.0 + np.exp(-eta))
            v = np.clip(mu * (1 - mu), 1e-10, None)
            z = eta + (p - mu) / v
            W = nn * v
        else:
            e = np.exp(np.clip(eta, -30, 30))
            mu = np.clip(1.0 - np.exp(-e), 1e-12, 1 - 1e-12)
            d = np.clip(e * np.exp(-e), 1e-12, None)
            v = np.clip(mu * (1 - mu), 1e-12, None)
            z = eta + (p - mu) / d
            W = nn * (d ** 2) / v
        A = X.T @ (X * W[:, None]) + ridge * np.eye(k)
        nb = np.linalg.solve(A, X.T @ (W * z))
        if not np.all(np.isfinite(nb)):
            return b, False
        if np.max(np.abs(nb - b)) < tol:
            return nb, True
        b = nb
    return b, True


def block(df, tag):
    vc = df["state"].value_counts()
    keep = set(vc[vc >= 40].index)
    df = df.copy()
    df["st"] = df["state"].where(df["state"].isin(keep), "other")
    agg, nrows = expand(df)
    ev = agg["ev"].to_numpy(float); n = agg["n"].to_numpy(float)
    Xf, nf = design(agg, True)
    Xn, nn_ = design(agg, False)
    b_no, _ = irls(Xn, ev, n)
    b_fe, _ = irls(Xf, ev, n)
    b_cl, _ = irls(Xf, ev, n, link="cloglog", start=b_fe * 0.5)

    rng = np.random.RandomState(SEED)
    sts = agg["st"].to_numpy(); uniq, inv = np.unique(sts, return_inverse=True)
    nS = len(uniq); draws = []; fails = 0
    for _ in range(NBOOT):
        cnt = np.bincount(rng.randint(0, nS, nS), minlength=nS).astype(float)
        bb, ok = irls(Xf, ev, n, w=cnt[inv], start=b_fe, maxit=12)
        draws.append(bb) if ok else None
        fails += (0 if ok else 1)
    D = np.array(draws)
    out = {"person_periods": nrows, "cells": int(len(agg)),
           "nboot_ok": int(len(draws)), "nboot_fail": int(fails),
           "or_no_fe": {c: float(np.exp(b_no[nn_.index(c)])) for c in HAZ_COVS},
           "or_fe": {c: float(np.exp(b_fe[nf.index(c)])) for c in HAZ_COVS},
           "hr_cloglog_fe": {c: float(np.exp(b_cl[nf.index(c)])) for c in HAZ_COVS},
           "or_fe_ci": {c: [float(np.percentile(np.exp(D[:, nf.index(c)]), 2.5)),
                            float(np.percentile(np.exp(D[:, nf.index(c)]), 97.5))]
                        for c in HAZ_COVS},
           "marginal_followup_rate": float(df["got_followup"].mean())}
    print("%-22s person-periods %d -> %d cells | boot ok %d fail %d"
          % (tag, nrows, len(agg), len(draws), fails))
    return out


res = {}
base = pd.read_parquet(DB / "events.parquet")
res["event_level"] = block(base, "event-level")
fa = DB / "events_firstart.parquet"
if fa.exists():
    res["first_article"] = block(pd.read_parquet(fa), "first-article")

json.dump(res, open(DB / "hazard_fe.json", "w"), indent=1)

print()
print("%-18s %9s %9s %18s %10s" % ("covariate", "OR no-FE", "OR +FE", "95% CI (+FE)", "cloglog"))
e = res["event_level"]
for c in HAZ_COVS:
    ci = e["or_fe_ci"][c]
    print("  %-16s %8.3f %8.3f  [%.2f, %.2f] %9.3f"
          % (c, e["or_no_fe"][c], e["or_fe"][c], ci[0], ci[1], e["hr_cloglog_fe"][c]))
if "first_article" in res:
    print()
    print("first-article predictors, +FE:")
    f = res["first_article"]
    for c in HAZ_COVS:
        ci = f["or_fe_ci"][c]
        print("  %-16s %8.3f  [%.2f, %.2f]" % (c, f["or_fe"][c], ci[0], ci[1]))
print("wrote", DB / "hazard_fe.json")
