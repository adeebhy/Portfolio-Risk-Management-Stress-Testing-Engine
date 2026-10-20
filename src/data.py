"""Market data: live yfinance fetch, CSV upload, and a labelled SYNTHETIC demo panel."""
import numpy as np, pandas as pd

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "EEM", "TLT", "LQD", "HYG", "GLD", "XLE", "XLK"]
# asset metadata used by stress tests / liquidity (durations approximate, editable)
META = {
 "SPY": dict(cls="Equity", sector="Broad", duration=0, adv=25e9, spread=0.0001),
 "QQQ": dict(cls="Equity", sector="Tech", duration=0, adv=12e9, spread=0.0001),
 "IWM": dict(cls="Equity", sector="Small cap", duration=0, adv=3.5e9, spread=0.0002),
 "EEM": dict(cls="Equity", sector="EM", duration=0, adv=1.2e9, spread=0.0004),
 "TLT": dict(cls="Bond", sector="Govt", duration=17.0, adv=2.5e9, spread=0.0002),
 "LQD": dict(cls="Bond", sector="IG credit", duration=8.5, adv=1.5e9, spread=0.0005),
 "HYG": dict(cls="Bond", sector="HY credit", duration=3.5, adv=1.8e9, spread=0.0008),
 "GLD": dict(cls="Commodity", sector="Gold", duration=0, adv=1.5e9, spread=0.0003),
 "XLE": dict(cls="Equity", sector="Energy", duration=0, adv=1.0e9, spread=0.0004),
 "XLK": dict(cls="Equity", sector="Tech", duration=0, adv=1.3e9, spread=0.0003),
}

def fetch_yfinance(tickers, years=10):
    """Live fetch. Returns (prices, volumes_usd) or raises. Blocked from the build sandbox
    (Yahoo returns 403 to its IP), so this path is real code but untested end-to-end there."""
    import yfinance as yf
    raw = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all").ffill().dropna()
    vol = (raw["Volume"] * raw["Close"]).tail(60).mean()   # 60d avg dollar volume
    return prices, vol

def load_csv(file) -> pd.DataFrame:
    """CSV with a Date column and one adjusted-close column per ticker."""
    df = pd.read_csv(file, index_col=0, parse_dates=True).sort_index()
    return df.dropna(how="all").ffill().dropna()

def synthetic_prices(tickers=DEFAULT_TICKERS, n_days=2520, seed=7):
    """SYNTHETIC demo data (NOT real prices): one-factor model, Student-t(4) shocks,
    volatility regimes, and a crisis window where correlations spike -- built so the
    VaR methods have fat tails and correlation breakdown to be tested against."""
    rng = np.random.default_rng(seed)
    beta = dict(SPY=1.0, QQQ=1.25, IWM=1.2, EEM=1.1, TLT=-0.25, LQD=0.15, HYG=0.45, GLD=0.0, XLE=1.0, XLK=1.3)
    idio = dict(SPY=.002, QQQ=.004, IWM=.006, EEM=.008, TLT=.007, LQD=.003, HYG=.003, GLD=.008, XLE=.011, XLK=.005)
    reg = np.where(rng.random(n_days) < 0.1, 2.5, 1.0)            # clustered high-vol regime
    for i in range(1, n_days):
        if rng.random() < 0.9: reg[i] = reg[i-1]
    crisis = slice(1100, 1180); reg[crisis] = 3.5
    t = lambda size: rng.standard_t(4, size) / np.sqrt(2.0)       # unit-variance t(4)
    mkt = 0.0004 + 0.0085 * reg * t(n_days)
    rets = {}
    for k in tickers:
        b = beta.get(k, 0.8)
        iv = idio.get(k, .006) * np.where(np.arange(n_days) >= 0, 1, 1)
        r = 0.0002 + b * mkt + iv * t(n_days)
        r[crisis] = r[crisis] + 0.6 * np.sign(b if b else 1) * 0.0 + (0.01 * b) * t(80)  # extra co-movement
        rets[k] = r
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(pd.DataFrame(rets, index=idx), axis=0)), index=idx)
    vol = pd.Series({k: META.get(k, {}).get("adv", 1e9) for k in tickers})
    return prices, vol
