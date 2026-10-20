# Portfolio Risk Management & Stress-Testing Engine

Market-risk engine: VaR and Expected Shortfall three ways, a VaR backtest, macro stress tests,
correlation/concentration analysis and liquidity analytics, behind an interactive Streamlit + Plotly dashboard.

```
pip install -r requirements.txt
python3 tests.py          # validation checks
streamlit run app.py
```

## What it does
| Module | Detail |
|---|---|
| **VaR / ES** | Historical simulation, parametric (variance-covariance), Monte Carlo (20k-200k runs, normal or Student-t via Cholesky). 90-99% confidence, 1-20 day horizon |
| **Backtest** | Rolling 250-day VaR vs. realised losses with the Kupiec proportion-of-failures test |
| **Stress tests** | Historical scenarios (2008, 2020, 2022) and custom shocks: rate +/- bps via modified duration, credit-spread widening, broad equity shock, sector shocks (e.g. Tech -15%) |
| **Correlation & concentration** | Correlation heatmap, risk contribution vs. capital weight, effective-N (1/HHI), flags pairs with correlation > 0.85 |
| **Liquidity** | Days-to-liquidate heatmap (asset x participation rate), half-spread cost, simplified liquidity-adjusted VaR |

## Validation (`tests.py`)
| Check | Result |
|---|---|
| Gaussian returns: three methods must agree | 99% VaR 1.80% / 1.81% / 1.81% (historical / parametric / MC) - agree |
| Fat-tailed returns: normal model should understate tail loss | Parametric ES 2.19% vs historical ES 3.81% (~43% understated); MC with Student-t recovers part of it (2.87%) |
| 100% TLT, +200 bps shock | -$340,000 on $1M (= -17 duration x 2%) - exact |
| Risk contributions sum to 100% | 1.0 |
| Kupiec backtest, 99% | **Rejected** (1.67% breach rate vs 1% expected, p=0.003): a 250-day window adapts too slowly to volatility clustering |
| Kupiec backtest, 95% | Not rejected (5.51%, p=0.27) |

The rejection is a real finding, not a bug: it is the standard argument for volatility-scaled (EWMA/GARCH/filtered historical) VaR.

## Honest limitations
1. **Demo data is synthetic.** Yahoo Finance returns 403 to this build environment's IP, so the live `yfinance` path (`src/data.py`) is real code
   but was **not tested end-to-end here**. The built-in demo panel is simulated (Student-t shocks, volatility regimes, a crisis window) and is labelled as such
   in the UI. Use "Live (yfinance)" or "Upload CSV" on your own machine for real analysis. The validation numbers above are on synthetic data.
2. **Historical scenario shocks are hand-entered, rounded approximations** of well-known drawdowns (not computed from price files). Verify before relying on them.
3. **Rate shocks use modified duration only** (no convexity, no curve shape); duration, ADV and spread values in `META` are static approximations - edit them.
4. **Square-root-of-time scaling** assumes iid returns; understates multi-day risk under volatility clustering.
5. **Liquidity VaR is simplified** (horizon scaling + half-spread). ADV is a 60-day average of dollar volume when fetched live, a fixed table otherwise.
6. MC uses the sample covariance (estimation error grows with many assets/short histories); no shrinkage applied.
7. Risk analytics, not investment advice.
