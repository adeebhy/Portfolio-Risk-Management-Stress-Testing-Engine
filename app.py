"""Portfolio Risk Management & Stress-Testing Engine. Run: streamlit run app.py"""
import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
from src.data import DEFAULT_TICKERS, META, fetch_yfinance, load_csv, synthetic_prices
from src.risk import *

st.set_page_config(page_title="Portfolio Risk Engine", layout="wide")
st.title("Portfolio Risk Management & Stress-Testing Engine")

# ---------------- Data ----------------
src = st.sidebar.radio("Data source", ["Live (yfinance)", "Upload CSV", "Synthetic demo"], index=2)
tickers = [t.strip().upper() for t in st.sidebar.text_input("Tickers", ",".join(DEFAULT_TICKERS)).split(",") if t.strip()]
prices = adv = None
if src == "Live (yfinance)":
    years = st.sidebar.slider("History (years)", 2, 20, 10)
    try:
        prices, adv = fetch_yfinance(tickers, years); st.sidebar.success(f"Fetched {len(prices)} days")
    except Exception as e:
        st.sidebar.error(f"yfinance failed: {e}. Use Upload CSV or the Synthetic demo.")
elif src == "Upload CSV":
    f = st.sidebar.file_uploader("CSV: Date + adjusted close per ticker", type="csv")
    if f is not None:
        prices = load_csv(f); tickers = list(prices.columns); adv = pd.Series({t: META.get(t, {}).get("adv", 1e9) for t in tickers})
else:
    prices, adv = synthetic_prices(tickers); st.warning("SYNTHETIC demo data (fat tails + a crisis window) - NOT real prices.")
if prices is None or prices.empty: st.stop()
tickers = [t for t in tickers if t in prices.columns]; prices = prices[tickers]
rets = np.log(prices).diff().dropna()

# ---------------- Portfolio ----------------
value = st.sidebar.number_input("Portfolio value ($)", 1e4, 1e10, 10_000_000.0, step=1e6)
wdf = st.sidebar.data_editor(pd.DataFrame({"ticker": tickers, "weight_%": [round(100 / len(tickers), 1)] * len(tickers)}), hide_index=True)
w = wdf["weight_%"].values / wdf["weight_%"].sum(); weights = dict(zip(wdf.ticker, w))
conf = st.sidebar.select_slider("Confidence", [0.90, 0.95, 0.975, 0.99], 0.99); horizon = st.sidebar.slider("Horizon (days)", 1, 20, 1)
n_sims = st.sidebar.slider("Monte Carlo simulations", 10_000, 200_000, 50_000, 10_000); dist = st.sidebar.radio("MC distribution", ["normal", "t"])

t1, t2, t3, t4, t5 = st.tabs(["VaR & ES", "Backtest", "Stress Tests", "Correlation & Concentration", "Liquidity"])
with t1:
    hv, he = historical_var_es(rets, w, conf, horizon); pv, pe = parametric_var_es(rets, w, conf, horizon)
    mv, me, sims = monte_carlo_var_es(rets, w, conf, horizon, n_sims, dist)
    tbl = pd.DataFrame({"Method": ["Historical", "Parametric", f"Monte Carlo ({dist}, {n_sims:,})"], "VaR %": [hv, pv, mv], "ES (CVaR) %": [he, pe, me]})
    tbl["VaR $"] = tbl["VaR %"] * value; tbl["ES $"] = tbl["ES (CVaR) %"] * value
    st.dataframe(tbl.style.format({"VaR %": "{:.2%}", "ES (CVaR) %": "{:.2%}", "VaR $": "${:,.0f}", "ES $": "${:,.0f}"}), width="stretch")
    fig = px.histogram(x=port_returns(rets, w) * np.sqrt(horizon), nbins=120, title="Historical portfolio return distribution")
    for name, v, c in [("Hist VaR", hv, "red"), ("Param VaR", pv, "orange"), ("MC VaR", mv, "purple"), ("Hist ES", he, "black")]:
        fig.add_vline(x=-v, line_dash="dash", line_color=c, annotation_text=name)
    st.plotly_chart(fig, width="stretch")
    if he > 1.25 * pe: st.warning(f"Historical ES is {he/pe-1:.0%} above parametric ES: returns are fat-tailed, so the normal model understates tail loss.")
with t2:
    st.caption("Rolling 250-day 1-day historical VaR vs. realised losses; Kupiec proportion-of-failures test.")
    k = kupiec_pof(rets, w, conf); st.json(k)
    (st.error if "REJECTED" in k["verdict"] else st.success)(k["verdict"])
with t3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Historical crash scenarios (approximate)")
        name = st.selectbox("Scenario", list(HIST_SCENARIOS)); df, pnl = historical_stress(weights, value, HIST_SCENARIOS[name])
        st.metric("Portfolio P&L", f"${pnl:,.0f}", f"{pnl/value:.1%}"); st.plotly_chart(px.bar(df, x="ticker", y="pnl", title=name), width="stretch")
    with c2:
        st.subheader("Custom shock")
        bps = st.slider("Rate shock (bps)", -300, 300, 200, 25); cs = st.slider("Credit spread widening (bps)", 0, 500, 0, 25)
        eq = st.slider("Broad equity shock (%)", -50, 20, 0) / 100; tech = st.slider("Tech sector extra shock (%)", -40, 0, -15) / 100
        df2, pnl2 = custom_stress(weights, value, bps, {"Tech": tech}, eq, cs)
        st.metric("Portfolio P&L", f"${pnl2:,.0f}", f"{pnl2/value:.1%}"); st.plotly_chart(px.bar(df2, x="ticker", y="pnl"), width="stretch")
    st.caption("Scenario shocks are hand-entered, rounded approximations of well-known drawdowns - verify before relying on them. Rate shocks use modified duration only (no convexity).")
with t4:
    corr = rets.corr(); st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap"), width="stretch")
    rc = risk_contributions(rets, w); cmp = pd.DataFrame({"weight": w, "risk_contribution": rc.values}, index=tickers)
    st.plotly_chart(px.bar(cmp.reset_index().melt("index"), x="index", y="value", color="variable", barmode="group", title="Capital weight vs. share of portfolio variance"), width="stretch")
    st.metric("Effective number of positions (1/HHI)", f"{effective_n(w):.1f} of {len(w)}")
    hot = [(a, b, corr.loc[a, b]) for i, a in enumerate(tickers) for b in tickers[i + 1:] if corr.loc[a, b] > 0.85]
    if hot: st.warning("Hidden concentration - highly correlated pairs (>0.85): " + ", ".join(f"{a}/{b} ({c:.2f})" for a, b, c in hot))
with t5:
    part = st.slider("Max participation of daily volume (%)", 5, 50, 20, 5) / 100; liq = liquidity_table(weights, value, adv, part)
    st.dataframe(liq.style.format({"position": "${:,.0f}", "adv_usd": "${:,.0f}", "days_to_liquidate": "{:.3f}", "half_spread_cost": "${:,.0f}"}), width="stretch")
    grid = pd.DataFrame({f"{p:.0%}": [weights[t] * value / (p * float(adv.get(t, 1e9))) for t in tickers] for p in (.05, .1, .2, .3, .5)}, index=tickers)
    st.plotly_chart(px.imshow(grid, text_auto=".3f", color_continuous_scale="YlOrRd", title="Days to liquidate: asset x participation rate"), width="stretch")
    lv, wd = liquidity_adjusted_var(hv, value, liq); st.metric("Liquidity-adjusted VaR (simplified)", f"${lv:,.0f}", f"vs ${hv*value:,.0f} plain VaR")
    st.caption("Large ETFs liquidate in a fraction of a day at this size - raise portfolio value or add illiquid names to see the heatmap light up. LVaR is a simplified horizon-scaling + half-spread approach.")
