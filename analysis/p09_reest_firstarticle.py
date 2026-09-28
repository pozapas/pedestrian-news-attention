"""p09_reest_firstarticle.py -- Review #5, estimation stage.

Refits the intensity model on predictors rebuilt from the first linked article
and reports it beside the consolidated-predictor fit, so the movement
attributable to outcome-dependent ascertainment is visible rather than argued.

Writes data_build/firstart_models.json.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ztnb  # noqa: E402

ROOT = HERE.parent
DB = ROOT / "data_build"
NBOOT = 2000
SEED = 2027
STATE_MIN = 40

MAIN = ["child", "teen", "elderly", "age_missing", "female", "gender_missing",
        "victim_named", "hit_run_flag", "hit_run_missing", "multi_fatality",
        "weekend"]


def collapse(df):
    vc = df["state"].value_counts()
    keep = set(vc[vc >= STATE_MIN].index)
    return df["state"].where(df["state"].isin(keep), "other")


def design(d, st):
    cols = [np.ones(len(d))]; names = ["const"]
    for c in MAIN:
        cols.append(d[c].to_numpy(float)); names.append(c)
    for lev in sorted(st.unique())[1:]:
        cols.append((st == lev).to_numpy(float)); names.append("st_" + lev)
    for lev in sorted(d["year"].unique())[1:]:
        cols.append((d["year"] == lev).to_numpy(float)); names.append("yr_%s" % lev)
    return np.column_stack(cols), names


def fit_beta(y, X, alpha, start=None):
    lna = np.log(alpha); k = X.shape[1]
    f = lambda b: ztnb.nll(np.r_[b, lna], y, X)
    g = lambda b: ztnb.grad(np.r_[b, lna], y, X)[:k]
    b0 = start if start is not None else np.r_[np.log(max(y.mean(), 1.01)), np.zeros(k - 1)]
    r = optimize.minimize(f, b0, jac=g, method="L-BFGS-B",
                          options=dict(maxiter=400, ftol=1e-12))
    return r.x, bool(r.success)


def run(df, tag, alpha=None):
    st = collapse(df)
    y = df["n_articles"].to_numpy(float)
    X, names = design(df, st)
    if alpha is None:
        cut = np.quantile(y, 0.99); keep = y <= cut
        def _j(lna):
            b, _ = fit_beta(y[keep], X[keep], float(np.exp(lna)))
            return ztnb.nll(np.r_[b, lna], y[keep], X[keep])
        gs = optimize.minimize_scalar(_j, bounds=(np.log(0.5), np.log(60)),
                                      method="bounded", options=dict(xatol=1e-4))
        alpha = float(np.exp(gs.x))
    beta, ok = fit_beta(y, X, alpha)
    rng = np.random.RandomState(SEED)
    sts = st.to_numpy(); uniq = np.unique(sts)
    idx = {s: np.flatnonzero(sts == s) for s in uniq}
    draws, fails = [], 0
    for _ in range(NBOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        bb, okb = fit_beta(y[rows], X[rows], alpha, start=beta)
        draws.append(bb) if okb else None
        fails += (0 if okb else 1)
    D = np.array(draws)
    out = {"tag": tag, "alpha": alpha, "converged": ok,
           "nboot_ok": int(len(draws)), "nboot_fail": int(fails),
           "rr": {}, "ci": {}}
    for c in MAIN:
        j = names.index(c)
        out["rr"][c] = float(np.exp(beta[j]))
        lo, hi = np.percentile(np.exp(D[:, j]), [2.5, 97.5])
        out["ci"][c] = [float(lo), float(hi)]
    return out


base = pd.read_parquet(DB / "events.parquet")
fa = pd.read_parquet(DB / "events_firstart.parquet")

res = {}
print("fitting consolidated-predictor model ...")
res["event_level"] = run(base, "event-level predictors")
print("  alpha=%.3f boot ok=%d" % (res["event_level"]["alpha"], res["event_level"]["nboot_ok"]))
print("fitting first-article-predictor model ...")
res["first_article"] = run(fa, "first-article predictors")
print("  alpha=%.3f boot ok=%d" % (res["first_article"]["alpha"], res["first_article"]["nboot_ok"]))

json.dump(res, open(DB / "firstart_models.json", "w"), indent=1)

print()
print("%-18s %22s %22s" % ("covariate", "event-level RR [CI]", "first-article RR [CI]"))
for c in MAIN:
    a, b = res["event_level"], res["first_article"]
    print("  %-16s %6.3f [%.2f,%.2f]   %6.3f [%.2f,%.2f]"
          % (c, a["rr"][c], a["ci"][c][0], a["ci"][c][1],
             b["rr"][c], b["ci"][c][0], b["ci"][c][1]))
print("wrote", DB / "firstart_models.json")
