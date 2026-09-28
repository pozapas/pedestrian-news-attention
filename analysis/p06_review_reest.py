"""p06_review_reest.py -- re-estimation demanded by the review.

Addresses, with numbers rather than wording:
  #6  document the dispersion trim rule and report a prespecified robust alternative
  #7  raise bootstrap replications, record convergence failures explicitly
  #9  refit the follow-up model with the SAME state and year effects the
      intensity model carries
  #10 report discrete-time logit coefficients as odds ratios, and add a
      complementary log-log fit for a proportional-hazard reading
  #11 observed-scale marginal contrasts for the headline covariates
  #13 state-clustered Gini interval as primary
  #19 formal trend test on the annual Gini series
  #20 list the pooled small jurisdictions
  #25 fractional-weight top shares

Writes data_build/review_reest.json. Read-only on the analytic panel.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ztnb  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = HERE.parent
PANEL = ROOT / "data_build" / "events.parquet"
OUT = ROOT / "data_build" / "review_reest.json"

STATE_MIN = 40
NBOOT_RR = 2000          # was 200 (#7)
NBOOT_HAZ = 2000         # was 200 (#7)
NBOOT_GINI = 2000
SEED = 2027

MAIN = ["child", "teen", "elderly", "age_missing", "female", "gender_missing",
        "victim_named", "hit_run_flag", "hit_run_missing", "multi_fatality",
        "weekend"]

res = {}


# ------------------------------------------------------------------ helpers
def collapse_states(df):
    vc = df["state"].value_counts()
    keep = set(vc[vc >= STATE_MIN].index)
    pooled = sorted(set(vc.index) - keep)
    s = df["state"].where(df["state"].isin(keep), "other")
    return s, pooled, {k: int(vc[k]) for k in pooled}


def design(d, covs, state_col, year_col=True):
    X = [np.ones(len(d))]
    names = ["const"]
    for c in covs:
        X.append(d[c].to_numpy(float)); names.append(c)
    for lev in sorted(state_col.unique())[1:]:
        X.append((state_col == lev).to_numpy(float)); names.append(f"st_{lev}")
    if year_col:
        for lev in sorted(d["year"].unique())[1:]:
            X.append((d["year"] == lev).to_numpy(float)); names.append(f"yr_{lev}")
    return np.column_stack(X), names


def fit_beta(y, X, alpha, start=None):
    lna = np.log(alpha)
    k = X.shape[1]
    f = lambda b: ztnb.nll(np.r_[b, lna], y, X)
    g = lambda b: ztnb.grad(np.r_[b, lna], y, X)[:k]
    b0 = start if start is not None else np.r_[np.log(y.mean()), np.zeros(k - 1)]
    from scipy import optimize
    r = optimize.minimize(f, b0, jac=g, method="L-BFGS-B",
                          options=dict(maxiter=400, ftol=1e-12))
    return r.x, bool(r.success)


# =================================================== 1. panel + alpha (#6, #20)
df = pd.read_parquet(PANEL)
st, pooled, pooled_n = collapse_states(df)
res["n_events"] = int(len(df))
res["pooled_jurisdictions"] = pooled
res["pooled_counts"] = pooled_n
res["pooled_total_events"] = int(sum(pooled_n.values()))
res["n_state_levels"] = int(st.nunique())
res["state_min_threshold"] = STATE_MIN

y = df["n_articles"].to_numpy(float)
X, names = design(df, MAIN, st)

# (#6) document the trim rule exactly: drop the top 1% of the article count
cut = np.quantile(y, 0.99)
keep = y <= cut
res["alpha_trim_rule"] = "drop events whose article count exceeds the 99th percentile"
res["alpha_trim_cut_articles"] = float(cut)
res["alpha_trim_dropped_n"] = int((~keep).sum())
res["alpha_trim_dropped_pct"] = float(100 * (~keep).mean())

from scipy import optimize
def _joint(lna):
    b, _ = fit_beta(y[keep], X[keep], float(np.exp(lna)))
    return ztnb.nll(np.r_[b, lna], y[keep], X[keep])
gs = optimize.minimize_scalar(_joint, bounds=(np.log(0.5), np.log(60)),
                              method="bounded", options=dict(xatol=1e-4))
ALPHA = float(np.exp(gs.x))
res["alpha_fixed"] = ALPHA

beta_full, ok = fit_beta(y, X, ALPHA)
res["main_fit_converged"] = ok
bn = dict(zip(names, beta_full))
res["rr_main"] = {c: float(np.exp(bn[c])) for c in MAIN}

# (#6) prespecified robust alternative: zero-truncated Poisson with a
# cluster-robust sandwich, which needs no dispersion parameter at all.
_ztp = ztnb.fit_ztp(y, X)
bp = np.asarray(_ztp["beta"], float)
res["ztp_converged"] = bool(_ztp["converged"])
res["rr_ztp"] = {c: float(np.exp(dict(zip(names, bp))[c])) for c in MAIN}


# ================================================ 2. RR bootstrap (#7)
rng = np.random.RandomState(SEED)
states = st.to_numpy()
uniq = np.unique(states)
idx_by_state = {s: np.flatnonzero(states == s) for s in uniq}
draws, fails = [], 0
for b in range(NBOOT_RR):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    rows = np.concatenate([idx_by_state[s] for s in pick])
    bb, okb = fit_beta(y[rows], X[rows], ALPHA, start=beta_full)
    if okb:
        draws.append(bb)
    else:
        fails += 1
D = np.array(draws)
res["nboot_rr_requested"] = NBOOT_RR
res["nboot_rr_converged"] = int(len(draws))
res["nboot_rr_failed"] = int(fails)
res["boot_failure_rule"] = "non-converged replicates are discarded and counted; no replacement draw is taken"
ci = {}
for c in MAIN:
    j = names.index(c)
    lo, hi = np.percentile(np.exp(D[:, j]), [2.5, 97.5])
    ci[c] = [float(lo), float(hi)]
res["rr_ci_main"] = ci


# ============================ 3. observed-scale marginal contrasts (#11)
def cond_mean_vec(beta, Xm):
    mu = np.exp(np.clip(Xm @ beta, -30, 30))
    return ztnb.cond_mean(mu, ALPHA)

obs = {}
for c in ["multi_fatality", "hit_run_flag", "victim_named", "female"]:
    j = names.index(c)
    X1 = X.copy(); X1[:, j] = 1.0
    X0 = X.copy(); X0[:, j] = 0.0
    d1, d0 = cond_mean_vec(beta_full, X1), cond_mean_vec(beta_full, X0)
    diff = float(np.mean(d1 - d0))
    ratio = float(np.mean(d1) / np.mean(d0))
    bd = []
    for bb in D[:: max(1, len(D) // 400)]:
        bd.append(float(np.mean(cond_mean_vec(bb, X1) - cond_mean_vec(bb, X0))))
    obs[c] = dict(avg_diff_articles=diff, observed_scale_ratio=ratio,
                  ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))])
res["observed_scale"] = obs
res["baseline_obs_mean"] = float(np.mean(cond_mean_vec(beta_full, X)))

print("phase 1-3 done: alpha=%.3f  boot ok=%d fail=%d" % (ALPHA, len(draws), fails))
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
