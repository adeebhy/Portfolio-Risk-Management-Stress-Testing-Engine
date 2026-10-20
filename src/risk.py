"""VaR / ES (historical, parametric, Monte Carlo), Kupiec backtest, stress tests,
risk contributions, and liquidity analytics. Losses are reported as POSITIVE numbers."""
import numpy as np, pandas as pd
from scipy import stats
from .data import META

# ---------------- VaR / Expected Shortfall ----------------
def port_returns(rets: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    return rets.values @ w

def historical_var_es(rets, w, conf=0.99, horizon=1):
    r = port_returns(rets, w)
    q = np.quantile(r, 1 - conf)
    var = -q; es = -r[r <= q].mean()
    s = np.sqrt(horizon)                       # square-root-of-time scaling (assumes iid)
    return var * s, es * s

def parametric_var_es(rets, w, conf=0.99, horizon=1):
    mu = float(rets.mean().values @ w); sig = float(np.sqrt(w @ rets.cov().values @ w))
    z = stats.norm.ppf(1 - conf)
    var = -(mu * horizon + z * sig * np.sqrt(horizon))
    es = -(mu * horizon - sig * np.sqrt(horizon) * stats.norm.pdf(z) / (1 - conf))
    return var, es

def monte_carlo_var_es(rets, w, conf=0.99, horizon=1, n_sims=20000, dist="normal", df_t=5, seed=42):
    """Correlated draws via Cholesky of the sample covariance. dist='t' uses a multivariate
    Student-t scaled to the same covariance (fat tails). Horizon applied by scaling mean/cov."""
    rng = np.random.default_rng(seed)
    mu = rets.mean().values * horizon; cov = rets.cov().values * horizon
    L = np.linalg.cholesky(cov + 1e-14 * np.eye(len(mu)))
    z = rng.standard_normal((n_sims, len(mu)))
    if dist == "t":
        z = z * np.sqrt((df_t - 2) / rng.chisquare(df_t, (n_sims, 1)))
    sims = mu + z @ L.T
    r = sims @ w
    q = np.quantile(r, 1 - conf)
    return -q, -r[r <= q].mean(), r

def kupiec_pof(rets, w, conf=0.99, window=250):
    """Rolling 1-day historical VaR backtest + Kupiec proportion-of-failures test."""
    r = port_returns(rets, w); T = len(r) - window; x = 0
    for i in range(window, len(r)):
        if r[i] < np.quantile(r[i - window:i], 1 - conf): x += 1
    p = 1 - conf; phat = x / T
    if x == 0: lr = -2 * T * np.log(1 - p)
    else: lr = -2 * ((T - x) * np.log(1 - p) + x * np.log(p) - (T - x) * np.log(1 - phat) - x * np.log(phat))
    return dict(observations=T, exceptions=x, expected=round(p * T, 1),
                exception_rate=round(phat, 4), LR_pof=round(float(lr), 3),
                p_value=round(float(1 - stats.chi2.cdf(lr, 1)), 4),
                verdict="Model adequate (fail to reject)" if (1 - stats.chi2.cdf(lr, 1)) > 0.05 else "Model REJECTED (too many/few breaches)")

# ---------------- Stress testing ----------------
HIST_SCENARIOS = {   # approximate rounded peak-to-trough / calendar moves; hand-entered, verify before relying on them
 "2008 Financial Crisis (Oct07-Mar09)": dict(SPY=-.55, QQQ=-.52, IWM=-.58, EEM=-.62, TLT=.22, LQD=-.12, HYG=-.28, GLD=.20, XLE=-.45, XLK=-.52),
 "2020 COVID Crash (Feb-Mar 2020)":     dict(SPY=-.34, QQQ=-.27, IWM=-.41, EEM=-.34, TLT=.15, LQD=-.17, HYG=-.22, GLD=-.03, XLE=-.58, XLK=-.30),
 "2022 Rate Shock (calendar 2022)":     dict(SPY=-.18, QQQ=-.33, IWM=-.20, EEM=-.20, TLT=-.31, LQD=-.22, HYG=-.11, GLD=-.01, XLE=.65, XLK=-.29),
}

def historical_stress(weights: dict, value: float, scenario: dict):
    rows = [(t, w * value, scenario.get(t, np.nan), w * value * scenario.get(t, 0.0)) for t, w in weights.items()]
    df = pd.DataFrame(rows, columns=["ticker", "position", "shock", "pnl"])
    return df, df.pnl.sum()

def custom_stress(weights: dict, value: float, rate_bps=0, sector_shocks=None, equity_shock=0.0, credit_spread_bps=0):
    """Rate shock via modified duration (dP = -D*dy), credit-spread widening on credit ETFs,
    broad equity shock, and sector-specific overrides (e.g. {'Tech': -0.15})."""
    sector_shocks = sector_shocks or {}; rows = []
    for t, w in weights.items():
        m = META.get(t, dict(cls="Equity", sector="Broad", duration=0)); shock = 0.0
        shock += -m["duration"] * rate_bps / 1e4
        if m["sector"] in ("IG credit", "HY credit"):
            shock += -(m["duration"] if m["duration"] else 4) * credit_spread_bps / 1e4
        if m["cls"] == "Equity": shock += equity_shock
        shock += sector_shocks.get(m["sector"], 0.0)
        rows.append((t, w * value, shock, w * value * shock))
    df = pd.DataFrame(rows, columns=["ticker", "position", "shock", "pnl"])
    return df, df.pnl.sum()

# ---------------- Concentration / correlation ----------------
def risk_contributions(rets, w):
    cov = rets.cov().values; sig = np.sqrt(w @ cov @ w)
    contrib = w * (cov @ w) / sig**2           # share of portfolio variance; sums to 1
    return pd.Series(contrib, index=rets.columns)

def effective_n(w): return 1 / np.sum(np.square(w))   # inverse HHI

# ---------------- Liquidity ----------------
def liquidity_table(weights: dict, value: float, adv: pd.Series, participation=0.2):
    rows = []
    for t, w in weights.items():
        pos = w * value; a = float(adv.get(t, 1e9)); days = pos / (participation * a)
        rows.append((t, pos, a, days, 0.5 * META.get(t, {}).get("spread", 0.001) * pos))
    return pd.DataFrame(rows, columns=["ticker", "position", "adv_usd", "days_to_liquidate", "half_spread_cost"])

def liquidity_adjusted_var(var_1d_frac, value, liq: pd.DataFrame):
    """Simplified LVaR: scale 1-day VaR by sqrt of the weighted liquidation horizon (min 1 day)
    and add exogenous half-spread cost. Illustrative, not a full Bangia/Jarrow model."""
    wdays = max(1.0, float((liq.days_to_liquidate * liq.position).sum() / liq.position.sum()))
    return var_1d_frac * value * np.sqrt(wdays) + liq.half_spread_cost.sum(), wdays
