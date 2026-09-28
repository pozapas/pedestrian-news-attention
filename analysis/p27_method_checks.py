"""p27_method_checks.py -- Method-section diagnostics under the single specification.

  ztp         zero-truncated Poisson fit of the main design (no dispersion
              parameter) and the likelihood-ratio statistic against it
  alpha_env   main-block rate ratios refit across alpha in [2.7, 20]
  recurrence  the recurrence model refit with 2,000 jurisdiction-bootstrap
              replications (matching every other interval in the paper), the
              complementary log-log agreement, and the gap between the
              model-implied and the directly computed 1 - S(t)

Writes data_build/method_checks.json.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gammaln

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ztnb  # noqa: E402
import p25_rebuild_models as R  # noqa: E402

DB = R.DB
fa = pd.read_parquet(DB / "events_firstart.parquet")
st = R.collapse(fa)
y = fa["n_articles"].to_numpy(float)
X, names = R.design(fa, st, R.MAIN, False)
out = {}

# ---- ZTP and LR
def ztp_nll(b):
    eta = np.clip(X @ b, -30, 30); mu = np.exp(eta)
    return -np.sum(y * eta - mu - gammaln(y + 1) - np.log(-np.expm1(-mu)))

def ztp_grad(b):
    eta = np.clip(X @ b, -30, 30); mu = np.exp(eta)
    return -(X.T @ (y - mu - mu * np.exp(-mu) / (-np.expm1(-mu))))

b0 = np.r_[np.log(max(y.mean() - 1, 0.1)), np.zeros(X.shape[1] - 1)]
rp = optimize.minimize(ztp_nll, b0, jac=ztp_grad, method="L-BFGS-B", options=dict(maxiter=1000, ftol=1e-13))
bnb, _ = R.fit(y, X)
ll_nb = -ztnb.nll(np.r_[bnb, np.log(R.ALPHA)], y, X)
ll_p = -rp.fun
out["ztp"] = {"converged": bool(rp.success),
              "rr": {c: float(np.exp(rp.x[names.index(c)])) for c in R.MAIN},
              "lr_stat": float(2 * (ll_nb - ll_p)), "ll_ztnb": float(ll_nb), "ll_ztp": float(ll_p)}

# ---- alpha envelope
env = {}
for a in [2.7, 3.5, 5.579314501383443, 8.0, 12.0, 20.0]:
    lna = np.log(a); k = X.shape[1]
    r = optimize.minimize(lambda b: ztnb.nll(np.r_[b, lna], y, X), bnb,
                          jac=lambda b: ztnb.grad(np.r_[b, lna], y, X)[:k],
                          method="L-BFGS-B", options=dict(maxiter=400, ftol=1e-12))
    mu = np.exp(np.clip(X @ r.x, -30, 30))
    env[str(round(a, 3))] = {"rr": {c: float(np.exp(r.x[names.index(c)])) for c in R.MAIN},
                             "mean_pred": float(ztnb.cond_mean(mu, a).mean())}
rng = {c: [min(v["rr"][c] for v in env.values()), max(v["rr"][c] for v in env.values())] for c in R.MAIN}
out["alpha_envelope"] = {"fits": env, "range": rng,
                         "rel_span": {c: (hi - lo) / ((hi + lo) / 2) for c, (lo, hi) in rng.items()}}

# ---- recurrence with 2,000 replications, cloglog, direct 1-S(t)
R.HAZ_NBOOT = 2000
rec, D = R.recurrence(fa)
fa2 = fa.copy(); fa2["st"] = R.collapse(fa2)
agg = R.expand(fa2)
ev = agg["ev"].to_numpy(float); n = agg["n"].to_numpy(float)
Xh, nh = R.haz_design(agg)
blog, _ = R.irls(Xh, ev, n)

def irls_cloglog(X, ev, n, start, maxit=60, tol=1e-9, ridge=1e-8):
    p = ev / np.maximum(n, 1e-12); b = start.copy(); k = X.shape[1]
    for _ in range(maxit):
        e = np.exp(np.clip(X @ b, -30, 30))
        mu = np.clip(1 - np.exp(-e), 1e-12, 1 - 1e-12)
        d = np.clip(e * np.exp(-e), 1e-12, None); v = mu * (1 - mu)
        z = np.log(e) + (p - mu) / d; W = n * d ** 2 / v
        nb = np.linalg.solve(X.T @ (X * W[:, None]) + ridge * np.eye(k), X.T @ (W * z))
        if np.max(np.abs(nb - b)) < tol:
            return nb
        b = nb
    return b

bcl = irls_cloglog(Xh, ev, n, blog * 0.5)
hr = {c: float(np.exp(bcl[nh.index(c)])) for c in R.HAZ_COVS}
# direct discrete-time survival from the person-period data
long_ev = agg.assign(t=agg["per"]).groupby("per")[["ev", "n"]].sum()
days = ["d%d" % t if t <= 7 else "w2" if t <= 14 else "w3" if t <= 21 else "w4" for t in range(1, 31)]
# per-day hazard for weekly bins spreads the bin's events over its at-risk person-days
h_emp = [long_ev.loc[p, "ev"] / long_ev.loc[p, "n"] for p in days]
S, emp = 1.0, []
for h in h_emp:
    S *= (1 - h); emp.append(1 - S)
# model curve, population averaged (from p25 recurrence standardization)
_, model_haz = None, rec["pop_hazard"]
# model-implied 1-S(t) population averaged is the cumulative of the event-level curves
out["recurrence"] = {"or": rec["or"], "ci": rec["ci"], "se_log": rec["se_log"], "nboot_ok": rec["nboot_ok"],
                     "cloglog_hr": hr,
                     "max_abs_or_minus_hr": max(abs(rec["or"][c] - hr[c]) for c in R.HAZ_COVS),
                     "cum_by_stratum": rec["cum_by_stratum"], "pop_hazard": rec["pop_hazard"],
                     "pop_cum30_model": rec["pop_cum30_model"],
                     "observed_followup_rate": rec["observed_followup_rate"],
                     "direct_cum": emp, "pop_cum": rec["pop_cum"],
                     "max_gap_direct_vs_model": float(max(abs(e - m) for e, m in zip(emp, rec["pop_cum"])))}
json.dump(out, open(DB / "method_checks.json", "w"), indent=1)

print("ZTP rate ratios:", {c: round(v, 2) for c, v in out["ztp"]["rr"].items()})
print("LR statistic ZTNB vs ZTP: %.0f" % out["ztp"]["lr_stat"])
print("alpha envelope ranges:", {c: [round(a, 3), round(b, 3)] for c, (a, b) in rng.items()})
print("alpha envelope relative span:", {c: round(v, 3) for c, v in out["alpha_envelope"]["rel_span"].items()})
print("mean prediction across alpha:", {k: round(v["mean_pred"], 4) for k, v in env.items()})
print("recurrence OR (2000 boots):", {c: (round(rec["or"][c], 2), [round(x, 2) for x in rec["ci"][c]]) for c in R.HAZ_COVS})
print("max |OR - cloglog HR| = %.3f" % out["recurrence"]["max_abs_or_minus_hr"])
print("direct 1-S(30) = %.4f ; model population 1-S(30) = %.4f ; max gap %.4f" % (emp[-1], rec["pop_cum30_model"], out["recurrence"]["max_gap_direct_vs_model"]))
