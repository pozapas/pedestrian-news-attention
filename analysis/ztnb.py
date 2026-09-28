"""ztnb.py -- zero-truncated negative binomial (NB2) MLE via scipy.

Implements the protocol §4.1 likelihood exactly and serves as the primary,
fully-controllable estimator (statsmodels' TruncatedLFNegativeBinomialP is used
separately as an agreement cross-check). Parameterization:

  mu_i = exp(x_i' beta),   r = 1/alpha,   alpha = exp(lnalpha) > 0
  NB2 pmf   f(y;mu,alpha) = C(y) (r/(r+mu))^r (mu/(r+mu))^y,  r=1/alpha
  f(0)      = (r/(r+mu))^r = (1+alpha*mu)^(-1/alpha)
  ZT pmf    P(Y=y | Y>=1) = f(y) / (1 - f(0)),   y >= 1
  cond mean E[Y | Y>=1]    = mu / (1 - f(0))

Analytic gradient provided and checked against finite differences in the unit test.
"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln, digamma
from scipy import optimize


def _unpack(theta, k):
    return theta[:k], theta[k]


def nll(theta, y, X):
    """Negative zero-truncated NB2 log-likelihood."""
    k = X.shape[1]
    beta, lnalpha = _unpack(theta, k)
    alpha = np.exp(lnalpha)
    r = 1.0 / alpha
    eta = X @ beta
    eta = np.clip(eta, -30, 30)
    mu = np.exp(eta)
    rpm = r + mu
    # log f(y)
    logf = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * (np.log(r) - np.log(rpm)) + y * (np.log(mu) - np.log(rpm)))
    # log(1 - f0);  f0 = exp(r*log(r/rpm))
    logf0 = r * (np.log(r) - np.log(rpm))
    f0 = np.exp(logf0)
    log_trunc = np.log1p(-f0)                      # log(1 - f0)
    ll = logf - log_trunc
    return -np.sum(ll)


def grad(theta, y, X):
    """Analytic gradient of nll."""
    k = X.shape[1]
    beta, lnalpha = _unpack(theta, k)
    alpha = np.exp(lnalpha)
    r = 1.0 / alpha
    eta = np.clip(X @ beta, -30, 30)
    mu = np.exp(eta)
    rpm = r + mu
    f0 = np.exp(r * (np.log(r) - np.log(rpm)))
    one_m_f0 = 1.0 - f0

    # d ll / d beta_j = r/rpm * [ (y - mu) - mu*f0/(1-f0) ] * x_j
    dbeta_factor = (r / rpm) * ((y - mu) - mu * f0 / one_m_f0)
    gbeta = X.T @ dbeta_factor

    # d ll / d r  (holding mu)
    dg_dr = (np.log(r) - np.log(rpm)) + 1.0 - r / rpm      # d/dr of r*log(r/rpm)
    dll_dr = (digamma(y + r) - digamma(r)
              + np.log(r) + 1.0 - np.log(rpm) - r / rpm - y / rpm
              + f0 * dg_dr / one_m_f0)
    # chain: dr/dlnalpha = -r
    dll_dlnalpha = dll_dr * (-r)
    glnalpha = np.sum(dll_dlnalpha)

    g = np.concatenate([gbeta, [glnalpha]])
    return -g


def poisson_init(y, X):
    """Quick Poisson IRLS-ish init for beta via least squares on log(y+0.5)."""
    lny = np.log(y + 0.5)
    beta, *_ = np.linalg.lstsq(X, lny, rcond=None)
    return beta


def fit(y, X, start=None, maxiter=500):
    """Fit ZT-NB2. Returns dict(beta, alpha, lnalpha, nll, converged, mu)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    k = X.shape[1]
    if start is None:
        b0 = poisson_init(y, X)
        start = np.concatenate([b0, [np.log(1.0)]])
    res = optimize.minimize(nll, start, args=(y, X), jac=grad,
                            method="BFGS", options={"maxiter": maxiter, "gtol": 1e-5})
    beta = res.x[:k]
    lnalpha = res.x[k]
    alpha = float(np.exp(lnalpha))
    eta = np.clip(X @ beta, -30, 30)
    mu = np.exp(eta)
    return dict(beta=beta, alpha=alpha, lnalpha=lnalpha, nll=float(res.fun),
                converged=bool(res.success), mu=mu, hess_inv=res.get("hess_inv"),
                res=res)


# -------------------- zero-truncated Poisson (for LR test) --------------------
def nll_ztp(beta, y, X):
    eta = np.clip(X @ beta, -30, 30)
    mu = np.exp(eta)
    ll = -mu + y * np.log(mu) - gammaln(y + 1.0) - np.log1p(-np.exp(-mu))
    return -np.sum(ll)

def grad_ztp(beta, y, X):
    eta = np.clip(X @ beta, -30, 30)
    mu = np.exp(eta)
    emm = np.exp(-mu)
    # d ll/d mu = -1 + y/mu - emm/(1-emm); dmu/dbeta = mu x
    fac = (-1.0 + y / mu - emm / (1.0 - emm)) * mu
    return -(X.T @ fac)

def fit_ztp(y, X, start=None, maxiter=500):
    y = np.asarray(y, float); X = np.asarray(X, float)
    if start is None:
        start = poisson_init(y, X)
    res = optimize.minimize(nll_ztp, start, args=(y, X), jac=grad_ztp,
                            method="BFGS", options={"maxiter": maxiter})
    return dict(beta=res.x, nll=float(res.fun), converged=bool(res.success))


def cond_mean(mu, alpha):
    """E[Y | Y>=1] = mu / (1 - f0)."""
    r = 1.0 / alpha
    f0 = (r / (r + mu)) ** r
    return mu / (1.0 - f0)


# --------------------------------- unit test ---------------------------------
def _simulate_ztnb(n, beta, alpha, seed):
    rng = np.random.RandomState(seed)
    k = len(beta)
    X = np.column_stack([np.ones(n)] + [rng.normal(0, 1, n) for _ in range(k - 1)])
    mu = np.exp(X @ beta)
    r = 1.0 / alpha
    lam = rng.gamma(shape=r, scale=mu / r)         # gamma-Poisson mixture = NB2
    y = rng.poisson(lam)
    keep = y >= 1                                   # zero-truncate
    return y[keep].astype(float), X[keep]


def unit_test(verbose=True):
    """Recover known beta/alpha from simulated ZTNB data; check gradient."""
    true_beta = np.array([0.4, 0.6, -0.3])
    true_alpha = 0.8
    y, X = _simulate_ztnb(40000, true_beta, true_alpha, seed=12345)

    # gradient check at a perturbed point
    theta0 = np.concatenate([true_beta * 0.5, [np.log(1.2)]])
    gnum = optimize.approx_fprime(theta0, nll, 1e-6, y, X)
    gana = grad(theta0, y, X)
    gerr = np.max(np.abs(gnum - gana) / (1 + np.abs(gnum)))

    out = fit(y, X)
    berr = np.max(np.abs(out["beta"] - true_beta))
    aerr = abs(out["alpha"] - true_alpha)
    ok = out["converged"] and berr < 0.05 and aerr < 0.08 and gerr < 1e-3
    if verbose:
        print(f"[ztnb unit test] converged={out['converged']} "
              f"beta_err={berr:.4f} alpha_err={aerr:.4f} grad_relerr={gerr:.2e}")
        print(f"    true beta={true_beta} alpha={true_alpha}")
        print(f"    est  beta={np.round(out['beta'],4)} alpha={out['alpha']:.4f}")
        print(f"    PASS={ok}")
    return ok


if __name__ == "__main__":
    unit_test()
