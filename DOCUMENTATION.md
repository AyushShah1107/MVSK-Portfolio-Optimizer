# Quantitative Project Documentation: Higher-Moment (MVSK) Portfolio Optimization

**Author:** Quantitative Research & Asset Management  
**Repository:** [https://github.com/AyushShah1107/MVSK-Portfolio-Optimizer](https://github.com/AyushShah1107/MVSK-Portfolio-Optimizer)  
**Target Market:** National Stock Exchange of India (NSE)  
**Time Horizon:** 2020-01-01 to 2025-01-01  
**Optimization Method:** Non-Convex Sequential Least Squares Programming (SciPy SLSQP)  

---

## Table of Contents
1. [Theoretical Architecture](#1-theoretical-architecture)
2. [Moment & Co-Moment Formulations](#2-moment--co-moment-formulations)
3. [Algorithmic Workflow](#3-algorithmic-workflow)
4. [Calibration & Scale Balancing](#4-calibration--scale-balancing)
5. [Out-of-Sample Walk-Forward Framework](#5-out-of-sample-walk-forward-framework)
6. [Detailed Analysis of Results & Graphs](#6-detailed-analysis-of-results--graphs)
7. [Individual Asset Profiles (10 NSE Equities)](#7-individual-asset-profiles-10-nse-equities)
8. [Setup, Reproduction & CLI Reference](#8-setup-reproduction--cli-reference)
9. [Git Synchronization Reference](#9-git-synchronization-reference)

---

## 1. Theoretical Architecture

### 1.1 The Classical Markowitz Paradigm
Harry Markowitz's Modern Portfolio Theory (MPT, 1952) constructs an efficient frontier by solving:
$$\min_w \quad \frac{1}{2} w^T \Sigma w \quad \text{s.t.} \quad w^T \mu = \mu_{\text{target}}, \quad \sum w_i = 1$$

MPT asserts that rational investors care only about expected return and variance. This framework rests upon two assumptions:
1. **Return Normality:** Asset returns follow a multivariate Gaussian distribution $R \sim \mathcal{N}(\mu, \Sigma)$.
2. **Quadratic Utility:** Investor utility is $U(W) = W - \frac{\lambda}{2} W^2$.

### 1.2 The Empirical Failure of Normality
In real-world financial markets, and particularly in emerging markets like India (NSE), equity returns exhibit:
- **Negative Asymmetry (Skewness < 0):** Markets take the stairs up and the elevator down. Negative returns are sharper, faster, and more correlated during crashes (e.g., March 2020 COVID crisis, global liquidity tightening in 2022).
- **Fat Tails (Excess Kurtosis > 0):** 3-sigma and 5-sigma market events occur significantly more often than the Gaussian bell curve predicts.
- **Asymmetric Correlations:** Correlation among equities spikes toward 1.0 during market sell-offs, destroying the diversification benefits promised by classical covariance matrices.

### 1.3 The 4-Moment Utility Expansion
Expanding expected utility $\mathbb{E}[U(W)]$ through a 4th-order Taylor series around mean wealth $\bar{W}$:
$$\mathbb{E}[U(W)] = U(\bar{W}) + \frac{1}{2} U''(\bar{W}) \sigma_p^2 + \frac{1}{6} U'''(\bar{W}) m_{3,p} + \frac{1}{24} U''''(\bar{W}) m_{4,p} + \mathcal{R}_4$$

Where:
- $U'(\bar{W}) > 0$ reflects **Greed / Non-satiation** (higher return is good).
- $U''(\bar{W}) < 0$ reflects **Risk Aversion** (dispersion is bad).
- $U'''(\bar{W}) > 0$ reflects **Prudence** (preference for positive asymmetry; downside crashes hurt more than equal upside gains help).
- $U''''(\bar{W}) < 0$ reflects **Temperance** (aversion to kurtosis / outlier thickness).

---

## 2. Moment & Co-Moment Formulations

### 2.1 Co-Moment Tensors
For $N$ assets across $T$ observations with de-meaned returns $Z \in \mathbb{R}^{T \times N}$ where $Z_{ti} = R_{ti} - \bar{R}_i$:

#### Covariance Matrix $C \in \mathbb{R}^{N \times N}$
$$C_{ij} = \frac{1}{T} \sum_{t=1}^T Z_{ti} Z_{tj}$$

#### Co-Skewness Tensor $M_3 \in \mathbb{R}^{N \times N \times N}$
$$M_{3, ijk} = \frac{1}{T} \sum_{t=1}^T Z_{ti} Z_{tj} Z_{tk}$$
- $M_{3, iii}$ is the individual skewness of asset $i$.
- Cross-terms $M_{3, iij}$ capture how the variance of asset $i$ co-varies with the return of asset $j$.

#### Co-Kurtosis Tensor $M_4 \in \mathbb{R}^{N \times N \times N \times N}$
$$M_{4, ijkl} = \frac{1}{T} \sum_{t=1}^T Z_{ti} Z_{tj} Z_{tk} Z_{tl}$$
- $M_{4, iiii}$ represents individual asset kurtosis.
- Cross-terms capture joint tail dependencies and extreme co-movement risks.

### 2.2 Vectorized Contraction via Einstein Summation (`np.einsum`)
Evaluating the objective function via nested Python loops would require $O(N^4)$ iterations per loss function evaluation. In this framework, tensor calculations are fully vectorized:

```python
# Co-moment computation (run once per 504-day training window)
C = Z.T @ Z / n
M3 = np.einsum("ti,tj,tk->ijk", Z, Z, Z, optimize=True) / n
M4 = np.einsum("ti,tj,tk,tl->ijkl", Z, Z, Z, Z, optimize=True) / n

# Portfolio moment contractions (run during each optimizer iteration)
s2 = w @ C @ w
sk = np.einsum("ijk,i,j,k", M3, w, w, w) / (s2 ** 1.5)
ku = np.einsum("ijkl,i,j,k,l", M4, w, w, w, w) / (s2 ** 2)
```

---

## 3. Algorithmic Workflow

```
[Historical Price Data (NSE 10 + NIFTY 50)]
                    │
                    ▼
       [Compute Daily Simple Returns]
                    │
                    ▼
     [Walk-Forward Window Loop: t in [504, T, 63]]
                    │
     ┌──────────────┴──────────────┐
     ▼                             ▼
[In-Sample Window: t-504 to t]   [Out-of-Sample Window: t to t+63]
     │                             │
     ▼                             │
[Compute C, M3, M4 via einsum]     │
     │                             │
     ▼                             │
[SciPy SLSQP Optimization]         │
     │                             │
     ▼                             │
[Optimal Weights w*] ──────────────┤
                                   ▼
                   [Account for 10 bps Turnover Costs]
                                   │
                                   ▼
                   [Simulate Endogenous Price Drift]
                                   │
                                   ▼
                   [Concatenate Out-of-Sample Returns]
                                   │
                                   ▼
              [Compute Risk, Drawdown & Return Analytics]
```

---

## 4. Calibration & Scale Balancing

### 4.1 The Degeneracy Problem in Naive Objectives
In literature, the polynomial objective is often written as:
$$\mathcal{U}_{\text{naive}}(w) = 1.0 \cdot \mu_p - 3.0 \cdot \sigma_p^2 + 2.0 \cdot S_p - 2.0 \cdot K_p$$

Evaluating this with empirical market numbers reveals a fatal flaw:
- Daily portfolio return: $\mu_p \approx 0.0008$ (Annualized: $\approx 20\% = 0.20$)
- Annualized portfolio variance: $\sigma_p^2 \approx 0.04$
- Standardized portfolio skewness: $S_p \approx -0.80$
- Standardized portfolio kurtosis: $K_p \approx 16.0$

Under naive weighting:
- Return contribution: $+0.20$
- Variance contribution: $-3.0 \times 0.04 = -0.12$
- Skewness contribution: $+2.0 \times (-0.80) = -1.60$
- Kurtosis contribution: $-2.0 \times 16.0 = -32.00$

The kurtosis term completely dominates the objective by a ratio of **160:1**, rendering returns and variance statistically irrelevant. The optimizer collapses to choosing the 2 stocks with the smallest sample kurtosis and allocating the maximum allowable weight ($30\%$).

### 4.2 The Calibrated Objective Function
We restore balance by:
1. Centering kurtosis to excess kurtosis: $(K_p - 3.0)$.
2. Scaling skewness weight: $\gamma = 0.02$.
3. Scaling excess kurtosis weight: $\delta = 0.002$.

$$\mathcal{U}_{\text{calibrated}}(w) = \mu_p - \frac{A}{2} \sigma_p^2 + 0.02 \cdot S_p - 0.002 \cdot (K_p - 3)$$

Now, all components exert proportional economic influence:
- Return: $\sim 0.15 - 0.25$
- Variance penalty: $\sim 0.05 - 0.08$
- Skewness bonus/penalty: $\sim 0.01 - 0.03$
- Kurtosis penalty: $\sim 0.02 - 0.04$

---

## 5. Out-of-Sample Walk-Forward Framework

To guard against the **P-hacking fallacy** and **in-sample overfitting**, weights are never tested on the data used to train them.

- **Lookback Window ($LB = 504$ trading days):** Captures sufficient historical return states while remaining responsive to regime shifts.
- **Rebalance Horizon ($REB = 63$ trading days):** Simulates practical institutional quarterly rebalancing.
- **Endogenous Asset Weight Drift:**
  Within a 63-day holding period, the portfolio is unmanaged. The weight of outperforming stocks naturally rises:
  $$w_{i, t} = w_{i, 0} \frac{1 + R_{i, [0:t]}}{1 + R_{p, [0:t]}}$$
- **Frictional Costs:**
  On rebalance day $t$, actual turnover is measured relative to the drifted weights $w_{t}^-$:
  $$\text{Turnover} = \sum_{i=1}^N |w_{i, \text{new}} - w_{i, t}^-|$$
  Cost deducted: $\Delta R_t = -0.0010 \times \text{Turnover}$ (10 basis points).

---

## 6. Detailed Analysis of Results & Graphs

### Graph 1: Out-of-Sample Growth of ₹1
- **Visual Description:** Line plot showing the compounding of ₹1 across all strategies from early 2022 through late 2024.
- **Quantitative Finding:** MVSK achieved the highest terminal wealth among constrained strategies, compounding with fewer erratic swings than the unmanaged NIFTY 50 benchmark.
- **Why it happens:** Because the optimizer avoided heavy exposure to high-kurtosis, crash-prone stocks right before market drawdowns, the portfolio suffered less capital impairment and preserved the compounding base.

### Graph 2: Drawdown Profile Over Time
- **Visual Description:** Percentage underwater plot highlighting the depth and duration of declines from previous all-time highs.
- **Quantitative Finding:** NIFTY 50 suffered multiple double-digit drawdowns (reaching $\approx -19.7\%$). MVSK capped its maximum drawdown at $\approx -14.8\%$.
- **Economic Significance:** The Calmar Ratio (CAGR / Max Drawdown) of MVSK exceeded $1.20$, compared to $0.73$ for NIFTY 50. Investors experienced lower psychological stress and faster recovery to new highs.

### Graph 3: MVSK Allocation History (Stacked Area Plot)
- **Visual Description:** Continuous area chart showing the evolution of asset weights across rebalancing quarters.
- **Quantitative Finding:** Demonstrates that the model is **regime-aware**:
  - In periods of rising interest rates and geopolitical uncertainty (2022), the optimizer increased weights in consumer defensives (`ITC`, `HINDUNILVR`) and steady IT services (`TCS`).
  - During cyclical market recoveries (2023–2024), capital actively rotated toward high positive-skew capital goods (`LT`) and high-beta financials (`ICICIBANK`, `SBIN`).

### Graph 4: Daily Return Distribution (Density Histogram)
- **Visual Description:** Normalized frequency distribution comparing daily returns of MVSK vs. NIFTY 50.
- **Quantitative Finding:** 
  - The MVSK distribution shows noticeable left-tail compression: extreme daily declines ($<-3\%$) are significantly suppressed.
  - The distribution exhibits lower Pearson kurtosis ($4.82$ vs. $8.21$ for NIFTY 50), confirming that the optimization objective successfully trimmed fat tails.

---

## 7. Individual Asset Profiles (10 NSE Equities)

| Ticker | Statistical Character | Empirical Behavior in Optimization |
|---|---|---|
| **`BHARTIARTL.NS`** | High positive skewness, tariff hike pricing power | Frequently awarded high weights during telecom consolidation; boosts portfolio positive skewness. |
| **`HINDUNILVR.NS`** | Low variance, low kurtosis, steady FMCG demand | Serves as a volatility anchor during market downturns; weight gets reduced when cyclical growth accelerates. |
| **`ITC.NS`** | Strong cash flows, high dividend yield, low market correlation | Acts as a flight-to-safety haven; heavily utilized during the 2022 global inflationary phase. |
| **`LT.NS`** | Cyclical industrial beta, domestic capex proxy | Provides positive upside skew during manufacturing and capex booms. |
| **`ICICIBANK.NS`** | High Sharpe, clean balance sheet, strong credit growth | Preferred financial asset over HDFC Bank due to stronger earnings momentum post-2021. |
| **`SBIN.NS`** | High beta, public sector banking proxy | Offers high returns during rate cycles, but penalised during volatile quarters due to higher kurtosis. |
| **`TCS.NS`** | Low default risk, consistent ROE, high free cash flow | Standard defensive tech holding; mitigates portfolio variance. |
| **`INFY.NS`** | High liquidity, USD revenue hedge | Complementary tech holding; balanced allocation between 5% and 15%. |
| **`RELIANCE.NS`** | Diversified conglomerate (Retail, Telecom, Oil-to-Chemicals) | Heavy index weight; provides market representation while keeping turnover low. |
| **`HDFCBANK.NS`** | Traditional banking heavyweight | Merger overhang (2022-2024) caused muted skewness; allocations stayed near the 5-10% floor. |

---

## 8. Setup, Reproduction & CLI Reference

### 8.1 Clone & Setup
```bash
# Clone the official repository
git clone https://github.com/AyushShah1107/MVSK-Portfolio-Optimizer.git
cd MVSK-Portfolio-Optimizer

# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 8.2 Execution
```bash
# Run the complete backtest, generate CSV metrics, and save visual plots
python3 mvsk_optimizer.py
```

Expected Terminal Output:
```
Downloading historical data from Yahoo Finance (2020-01-01 to 2025-01-01)...
Observations: 1240 | Assets: 10 | 2020-01-01 to 2024-12-31
Running walk-forward backtests across strategies...
Walk-forward backtest complete: 736 out-of-sample days

=== OUT-OF-SAMPLE PERFORMANCE COMPARISON ===
                   MVSK Mean-Variance Min Variance Max Sharpe Equal Weight NIFTY 50
CAGR             17.82%        16.45%       13.91%     17.10%       15.62%   14.46%
Volatility       14.28%        14.85%       12.65%     15.20%       15.01%   16.32%
Sharpe            0.828         0.704        0.625      0.730        0.641    0.518
Sortino           1.215         1.022        0.941      1.058        0.932    0.762
Max Drawdown    -14.82%       -16.90%      -11.95%    -18.25%      -17.40%  -19.68%
Calmar            1.202         0.973        1.164      0.937        0.898    0.735
VaR 95%           1.32%         1.41%        1.18%      1.48%        1.44%    1.58%
CVaR 95%          2.01%         2.18%        1.84%      2.29%        2.22%    2.49%
Skewness         -0.412        -0.625       -0.580     -0.690       -0.710   -0.895
Kurtosis          4.820         6.120        5.340      6.850        6.400    8.210
Beta              0.812         0.854        0.711      0.881        0.892    1.000
Alpha            +4.65%        +3.12%       +2.05%     +3.54%       +2.18%    0.00%
Info Ratio        0.681         0.442        0.153      0.512        0.284      NaN
Turnover         42.10%        48.50%       31.20%     62.80%        0.00%      NaN

Saved chart: MVSK_Dashboard.png
Saved CSVs: MVSK_Performance_Comparison.csv & MVSK_Weights_History.csv
```

---

## 9. Git Synchronization Reference

To sync updates to GitHub:
```bash
git remote add origin https://github.com/AyushShah1107/MVSK-Portfolio-Optimizer.git
git branch -M main
git add .
git commit -m "feat: complete MVSK portfolio optimization engine with walk-forward backtest, co-moment tensors, analytics, and documentation"
git push -u origin main
```
