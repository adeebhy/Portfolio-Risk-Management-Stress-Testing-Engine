import numpy as np, pandas as pd
from src.data import synthetic_prices
from src.risk import *
# 1) Gaussian data: all three methods must agree
rng = np.random.default_rng(1); n = 5000
C = np.array([[1,.5,.2],[.5,1,.3],[.2,.3,1]]) * 1e-4
g = pd.DataFrame(rng.multivariate_normal([0,0,0], C, n), columns=list("ABC")); w = np.array([.5,.3,.2])
h, p = historical_var_es(g, w, .99)[0], parametric_var_es(g, w, .99)[0]; m = monte_carlo_var_es(g, w, .99, n_sims=100000)[0]
print(f"[Gaussian] VaR99 hist={h:.4f} param={p:.4f} mc={m:.4f}"); assert max(h,p,m)/min(h,p,m) < 1.08
# 2) Fat-tailed synthetic panel: parametric ES should UNDERSTATE historical ES
prices, _ = synthetic_prices(); r = np.log(prices).diff().dropna(); w = np.ones(10)/10
hv, he = historical_var_es(r, w, .99); pv, pe = parametric_var_es(r, w, .99); tv, te, _ = monte_carlo_var_es(r, w, .99, dist="t")
print(f"[Fat tails] ES99 hist={he:.4f} param={pe:.4f} mc-t={te:.4f} | VaR99 hist={hv:.4f} param={pv:.4f} mc-t={tv:.4f}")
assert he > pe
print("[Kupiec 99%]", kupiec_pof(r, w, .99))
print("[Kupiec 95%]", kupiec_pof(r, w, .95))
# 3) Stress arithmetic: 100% TLT, +200bps -> -17*0.02 = -34%
df, pnl = custom_stress({"TLT":1.0}, 1e6, rate_bps=200); print("[Stress] TLT +200bps pnl:", pnl); assert abs(pnl + 340000) < 1
print("[Contrib sum]", round(risk_contributions(r, w).sum(), 6))
print("ALL TESTS PASSED")
