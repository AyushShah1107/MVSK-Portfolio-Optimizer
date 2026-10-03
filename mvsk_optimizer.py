"""
Higher-Moment (MVSK) Portfolio Optimization vs NIFTY 50
========================================================
Walk-forward, cost-aware backtest of a Mean-Variance-Skewness-Kurtosis allocation
across 10 large-cap NSE equities (2020-2025).

Assets:
- RELIANCE.NS, TCS.NS, HDFCBANK.NS, ICICIBANK.NS, INFY.NS,
  ITC.NS, LT.NS, SBIN.NS, BHARTIARTL.NS, HINDUNILVR.NS
Benchmark:
- ^NSEI (NIFTY 50)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS"
]
BENCH = "^NSEI"
START, END = "2020-01-01", "2025-01-01"
RF = 0.06       # 6.0% risk-free rate (India RBI repo/T-bill proxy)
TD = 252        # Trading days per year
LB = 504        # 2-year rolling lookback window (504 trading days)
REB = 63        # Quarterly rebalancing interval (63 trading days)
COST = 0.001    # 10 bps transaction cost per unit turnover
BOUNDS = (0.05, 0.30) # 5% minimum to 30% maximum allocation per asset
A = 3.0         # Risk aversion parameter
G = 0.02        # Skewness preference scaling weight
D = 0.002       # Excess kurtosis penalty scaling weight

print(f"Downloading historical data from Yahoo Finance ({START} to {END})...")
try:
    px = yf.download(TICKERS + [BENCH], start=START, end=END, auto_adjust=True, progress=False)["Close"].dropna()
except Exception as e:
    print(f"Error downloading data: {e}")
    sys.exit(1)

rets = px.pct_change().dropna()
R = rets[TICKERS]
B = rets[BENCH]
N = len(TICKERS)
print(f"Observations: {len(R)} | Assets: {N} | {R.index[0].date()} to {R.index[-1].date()}")

def moments(X):
    """
    Computes annualized mean return, sample covariance matrix C,
    sample co-skewness tensor M3 (N x N x N), and co-kurtosis tensor M4 (N x N x N x N)
    using efficient Einstein summation (np.einsum).
    """
    n = len(X)
    Z = X - X.mean(0)
    C = Z.T @ Z / n
    M3 = np.einsum("ti,tj,tk->ijk", Z, Z, Z, optimize=True) / n
    M4 = np.einsum("ti,tj,tk,tl->ijkl", Z, Z, Z, Z, optimize=True) / n
    return X.mean(0) * TD, C, M3, M4

def port_stats(w, m):
    """
    Calculates expected return, annualized variance, skewness, and kurtosis
    via tensor contractions with portfolio weight vector w.
    """
    mu, C, M3, M4 = m
    s2 = w @ C @ w
    sk = np.einsum("ijk,i,j,k", M3, w, w, w) / (s2 ** 1.5)
    ku = np.einsum("ijkl,i,j,k,l", M4, w, w, w, w) / (s2 ** 2)
    return w @ mu, s2 * TD, sk, ku

def fit(X, kind):
    """
    Optimizes portfolio weights using SciPy SLSQP subject to:
    - Full investment constraint: sum(w) == 1
    - Box bounds: 0.05 <= w_i <= 0.30
    """
    m = moments(X)
    def loss(w):
        r, v, sk, ku = port_stats(w, m)
        if kind == "mvsk":
            # Maximize: r - 0.5*A*v + G*sk - D*(ku - 3)
            return -(r - 0.5 * A * v + G * sk - D * (ku - 3))
        if kind == "mv":
            # Classical Mean-Variance Markowitz
            return -(r - 0.5 * A * v)
        if kind == "minvar":
            # Global Minimum Variance
            return v
        # Max Sharpe Ratio (annualized return - rf) / sqrt(variance)
        return -(r - RF) / np.sqrt(v)
    
    res = minimize(
        loss,
        np.full(N, 1.0 / N),
        method="SLSQP",
        bounds=[BOUNDS] * N,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-10}
    )
    return res.x

def backtest(kind):
    """
    Strict walk-forward backtest:
    - Train on [t - LB : t] (strictly out-of-sample forward evaluation)
    - Hold for REB trading days (weights drift with asset returns)
    - Deduct 10 bps transaction cost per unit of turnover at rebalance day
    """
    out, W, turn = [], [], []
    prev = np.full(N, 1.0 / N)
    for t in range(LB, len(R), REB):
        w = np.full(N, 1.0 / N) if kind == "ew" else fit(R.iloc[t - LB : t].values, kind)
        blk = R.iloc[t : t + REB]
        g = (1 + blk.values).cumprod(0)
        v = np.r_[1, g @ w]
        r = v[1:] / v[:-1] - 1
        turn.append(np.abs(w - prev).sum())
        r[0] -= COST * turn[-1]
        prev = w * g[-1] / (g[-1] @ w)
        out.append(pd.Series(r, index=blk.index))
        W.append(pd.Series(w, index=TICKERS, name=R.index[t]))
    return pd.concat(out), pd.DataFrame(W), np.mean(turn) * TD / REB

print("Running walk-forward backtests across strategies...")
names = {
    "mvsk": "MVSK",
    "mv": "Mean-Variance",
    "minvar": "Min Variance",
    "maxsharpe": "Max Sharpe",
    "ew": "Equal Weight"
}
res = {k: backtest(k) for k in names}
oos = res["mvsk"][0].index
bench = B.loc[oos]
print(f"Walk-forward backtest complete: {len(oos)} out-of-sample days ({oos[0].date()} to {oos[-1].date()})")

def evaluate(r, b, turn=np.nan):
    cum = (1 + r).cumprod()
    yrs = len(r) / TD
    vol = r.std() * np.sqrt(TD)
    dn = np.sqrt((np.minimum(r, 0) ** 2).mean() * TD)
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    mdd = (cum / cum.cummax() - 1).min()
    beta = np.cov(r, b)[0, 1] / b.var()
    alpha = (r.mean() - RF / TD - beta * (b.mean() - RF / TD)) * TD
    ex = r - b
    q = np.percentile(r, 5)
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": (r.mean() * TD - RF) / vol,
        "Sortino": (r.mean() * TD - RF) / dn if dn > 0 else np.nan,
        "Max Drawdown": mdd,
        "Calmar": cagr / abs(mdd) if abs(mdd) > 0 else np.nan,
        "VaR 95%": -q,
        "CVaR 95%": -r[r <= q].mean(),
        "Skewness": skew(r),
        "Kurtosis": kurtosis(r, fisher=False),
        "Beta": beta,
        "Alpha": alpha,
        "Info Ratio": ex.mean() * TD / (ex.std() * np.sqrt(TD)) if not r.equals(b) else np.nan,
        "Turnover": turn
    }

table = pd.DataFrame({names[k]: evaluate(res[k][0], bench, res[k][2]) for k in names})
table["NIFTY 50"] = evaluate(bench, bench)

pct = ["CAGR", "Volatility", "Max Drawdown", "VaR 95%", "CVaR 95%", "Alpha", "Turnover"]
show = table.copy().astype(object)
for i in table.index:
    show.loc[i] = [f"{v:.2%}" if i in pct else f"{v:.3f}" for v in table.loc[i]]

print("\n=== OUT-OF-SAMPLE PERFORMANCE COMPARISON ===")
print(show.to_string())

mvsk_in = port_stats(fit(R.values, "mvsk"), moments(R.values))
print("\nFull-sample MVSK (in-sample) | Ret %.3f Var %.4f Skew %.3f Kurt %.2f" % mvsk_in)

avgw = res["mvsk"][1].mean().round(4).rename("Avg MVSK Weight")
print("\n=== AVERAGE OUT-OF-SAMPLE ALLOCATION (MVSK) ===")
print(avgw.to_string())

print("\nGenerating Diagnostic Charts...")
cums = {names[k]: (1 + res[k][0]).cumprod() for k in names}
cums["NIFTY 50"] = (1 + bench).cumprod()

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for n_, s in cums.items():
    ax[0, 0].plot(s, label=n_, lw=2.4 if n_ == "MVSK" else 1.2, ls="--" if n_ == "NIFTY 50" else "-")
    ax[0, 1].plot(s / s.cummax() - 1, label=n_, lw=2.4 if n_ == "MVSK" else 1.2, ls="--" if n_ == "NIFTY 50" else "-")

ax[0, 0].set(title="Out-of-Sample Growth of ₹1", ylabel="Portfolio Wealth (₹)")
ax[0, 0].legend(fontsize=8)
ax[0, 1].set(title="Drawdown Profile", ylabel="Drawdown (%)")
ax[0, 1].legend(fontsize=8)

res["mvsk"][1].plot.area(ax=ax[1, 0], stacked=True, alpha=0.85, legend=False)
ax[1, 0].set(title="MVSK Weights at Each Rebalance", ylim=(0, 1), ylabel="Portfolio Weight")
ax[1, 0].legend(TICKERS, fontsize=6, ncol=2, loc="upper left")

ax[1, 1].hist(res["mvsk"][0], bins=70, alpha=0.6, density=True, label="MVSK", color="#2563eb")
ax[1, 1].hist(bench, bins=70, alpha=0.5, density=True, label="NIFTY 50", color="#dc2626")
ax[1, 1].set(title="Daily Return Distribution (MVSK vs NIFTY 50)", xlabel="Daily Return")
ax[1, 1].legend()

plt.tight_layout()
plt.savefig("MVSK_Dashboard.png", dpi=150)
print("Saved chart: MVSK_Dashboard.png")

table.to_csv("MVSK_Performance_Comparison.csv")
res["mvsk"][1].to_csv("MVSK_Weights_History.csv")
print("Saved CSVs: MVSK_Performance_Comparison.csv & MVSK_Weights_History.csv")
