# 📈 Monte Carlo Portfolio Risk Simulator

This project estimates the potential future performance and downside risk of a stock portfolio using a **Monte Carlo simulation** based on **Geometric Brownian Motion (GBM)**. To capture realistic market behavior, the simulation incorporates **correlation between assets** using **Cholesky decomposition**.

 ![Alt text](image.png)

---

## 🚀 Project Overview

This simulation models the 1-year future return distribution of a portfolio consisting of multiple stocks (e.g., AAPL, TSLA, NVDA, AMZN), accounting for:

- Historical mean returns and covariances
- Asset correlation
- 10,000 random future scenarios
- Portfolio-level aggregation of returns

The outcome is a full distribution of simulated portfolio values, from which we compute:
- ✅ **Value at Risk (VaR)** at the 5% level
- ✅ **Expected Shortfall (ES)**
- ✅ **Annual return distribution**
- ✅ **Visual histogram of potential outcomes**

---

## 📊 Key Results (Sample Run)

- **Initial Portfolio Value:** \$173.35  
- **Simulations:** 10,000  
- **Time Horizon:** 252 trading days  
- **Mean Simulated Return:** 108.06%  
- **Standard Deviation:** 70.96%  
- **5% Value at Risk (VaR):** 22.72%  
- **Expected Shortfall (ES):** 10.05%

---

## 🧪 How It Works

### 1. **Data Acquisition**
- Uses the `yfinance` API to download daily adjusted close prices.
- Tickers: customizable (e.g., `['AAPL', 'TSLA', 'NVDA', 'AMZN']`)

### 2. **Return Modeling**
- Computes daily log returns and estimates mean vector & covariance matrix.
- Correlation between assets is modeled using **Cholesky decomposition**.

### 3. **GBM Simulation**
- Simulates 252 trading days for each asset using:
  

- Shocks (`Z_t`) are correlated via Cholesky transformation.

### 4. **Portfolio Aggregation**
- Combines each simulation into a total portfolio return using weights.
- Final portfolio values used to estimate VaR and Expected Shortfall.

---

## 🛠 Libraries Used

- [`numpy`](https://numpy.org/)
- [`pandas`](https://pandas.pydata.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`scipy`](https://scipy.org/)
- [`yfinance`](https://github.com/ranaroussi/yfinance)

---
