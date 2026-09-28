"""p15_single_spec.py -- one specification for the whole paper.

Every covariate is an indicator for "the first linked article reported X".
There are no missing-category indicators anywhere, because such an indicator is
partly a consequence of coverage: an event carried by few articles offers few
opportunities to report any attribute, so conditioning on its missingness means
conditioning on a descendant of the outcome. Folding "not reported" into the
reference category removes that channel and gives every covariate the same
interpretation.

Fits the main block and the secondary (context) block on both the first-article
panel and, for comparison, the consolidated event record.

Writes data_build/single_spec.json.
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
ALPHA = 5.579314501383443
NBOOT = 2000
SEED = 2027
STATE_MIN = 40

MAIN = ["child", "teen", "elderly", "female", "victim_named",
        "hit_run_flag", "multi_fatality", "weekend"]
SEC = MAIN + ["dark", "charges_reported"]


def collapse(df):
    vc = df["state"].value_counts()
    keep = set(vc[vc >= STATE_MIN].index)
    return df["state"].where(df["state"].isin(keep), "other")


def design(d, st, covs, road=False):
    cols = [np.ones(len(d))]
    names = ["const"]
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
    r = optimize.minimize(f, b0, jac=g, method="L-BFGS-B",
                          options=dict(maxiter=400, ftol=1e-12))
    return r.x, bool(r.success)


def run(df, covs, road, report):
    st = collapse(df)
    y = df["n_articles"].to_numpy(float)
    X, names = design(df, st, covs, road)
    beta, ok = fit(y, X)
    rng = np.random.RandomState(SEED)
    sts = st.to_numpy(); uniq = np.unique(sts)
    idx = {s: np.flatnonzero(sts == s) for s in uniq}
    D, fails = [], 0
    for _ in range(NBOOT):
        rows = np.concatenate([idx[s] for s in rng.choice(uniq, size=len(uniq), replace=True)])
        bb, okb = fit(y[rows], X[rows], start=beta)
        D.append(bb) if okb else None
        fails += (0 if okb else 1)
    D = np.array(D)
    out = {"nboot_ok": int(len(D)), "nboot_fail": int(fails), "converged": ok,
           "rr": {}, "ci": {}}
    for c in report:
        j = names.index(c)
        out["rr"][c] = float(np.exp(beta[j]))
        out["ci"][c] = [float(np.percentile(np.exp(D[:, j]), 2.5)),
                        float(np.percentile(np.exp(D[:, j]), 97.5))]
    # observed-scale average contrasts
    def cm(b, Xm):
        return ztnb.cond_mean(np.exp(np.clip(Xm @ b, -30, 30)), ALPHA)
    out["obs_mean"] = float(cm(beta, X).mean())
    out["obs_diff"] = {}
    for c in report:
        j = names.index(c)
        X1 = X.copy(); X1[:, j] = 1.0
        X0 = X.copy(); X0[:, j] = 0.0
        out["obs_diff"][c] = float((cm(beta, X1) - cm(beta, X0)).mean())
    return out


res = {}
fa = pd.read_parquet(DB / "events_firstart.parquet")
base = pd.read_parquet(DB / "events.parquet")

res["first_article_main"] = run(fa, MAIN, False, MAIN)
print("first-article main block done")
res["event_level_main"] = run(base, MAIN, False, MAIN)
print("event-level main block done")
res["first_article_sec"] = run(fa, SEC, True, ["dark", "charges_reported"])
print("first-article secondary block done")
res["event_level_sec"] = run(base, SEC, True, ["dark", "charges_reported"])
print("event-level secondary block done")

json.dump(res, open(DB / "single_spec.json", "w"), indent=1)
print()
print("%-18s %24s %24s" % ("covariate", "first-article RR [CI]", "event-record RR [CI]"))
for c in MAIN:
    a, b = res["first_article_main"], res["event_level_main"]
    print("  %-16s %8.3f [%.2f, %.2f]      %8.3f [%.2f, %.2f]"
          % (c, a["rr"][c], a["ci"][c][0], a["ci"][c][1],
             b["rr"][c], b["ci"][c][0], b["ci"][c][1]))
for c in ["dark", "charges_reported"]:
    a, b = res["first_article_sec"], res["event_level_sec"]
    print("  %-16s %8.3f [%.2f, %.2f]      %8.3f [%.2f, %.2f]"
          % (c, a["rr"][c], a["ci"][c][0], a["ci"][c][1],
             b["rr"][c], b["ci"][c][0], b["ci"][c][1]))
print()
print("observed-scale contrasts (first-article main), mean %.3f:" % res["first_article_main"]["obs_mean"])
for c, v in res["first_article_main"]["obs_diff"].items():
    print("   %-16s %+.3f articles" % (c, v))
print("wrote", DB / "single_spec.json")
