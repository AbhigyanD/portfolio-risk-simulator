import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.linalg import cholesky

# Parameters
tickers = ['AAPL', 'TSLA', 'NVDA', 'AMZN']
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
start_date = "2023-06-25"
end_date = "2024-06-25"
n_simulations = 10_000
n_days = 252  # 1 trading year

# Fetch adjusted close prices
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
log_returns = np.log(data / data.shift(1)).dropna()

# Estimate mean and covariance
mu = log_returns.mean().values  # daily mean returns
cov = log_returns.cov().values  # daily covariance matrix

# Cholesky decomposition for correlation
L = cholesky(cov, lower=True)

# Initialize
initial_prices = data.iloc[-1].values
simulated_portfolios = np.zeros((n_simulations, n_days))

# Simulate paths
for i in range(n_simulations):
    correlated_randoms = np.dot(L, np.random.normal(size=(len(tickers), n_days)))
    price_paths = np.zeros((n_days, len(tickers)))
    price_paths[0] = initial_prices

    for t in range(1, n_days):
        drift = (mu - 0.5 * np.diag(cov))
        diffusion = correlated_randoms[:, t]
        price_paths[t] = price_paths[t-1] * np.exp(drift + diffusion)

    # Aggregate weighted portfolio value
    portfolio_path = np.dot(price_paths, weights)
    simulated_portfolios[i] = portfolio_path

# Analyze final values
final_portfolio_values = simulated_portfolios[:, -1]
initial_value = np.dot(initial_prices, weights)
returns = final_portfolio_values / initial_value - 1

# Value at Risk (5%)
VaR_95 = np.percentile(returns, 5)
ES_95 = returns[returns <= VaR_95].mean()

# Plot
plt.hist(returns, bins=100, color='skyblue', edgecolor='black')
plt.axvline(VaR_95, color='red', linestyle='--', label=f'5% VaR: {VaR_95:.2%}')
plt.axvline(ES_95, color='purple', linestyle='--', label=f'ES: {ES_95:.2%}')
plt.title('Monte Carlo Simulation: Portfolio Return Distribution (1 Year)')
plt.xlabel('Portfolio Return')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Print results
print(f"Initial Portfolio Value: ${initial_value:,.2f}")
print(f"5% Value at Risk (VaR): {VaR_95:.2%}")
print(f"Expected Shortfall (ES): {ES_95:.2%}")
