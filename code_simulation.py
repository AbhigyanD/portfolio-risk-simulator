import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.linalg import cholesky

# --- Parameters ---
tickers = ['AAPL', 'TSLA', 'NVDA', 'AMZN']
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
start_date = "2024-01-01"
end_date = "2024-06-25" # Or even datetime.now() for current date
n_simulations = 10_000
n_trading_days = 252  # Number of trading days for the simulation horizon (e.g., 1 year)

# --- Data Fetching and Preparation ---
print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
try:
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    if data.empty:
        raise ValueError("No data downloaded. Check tickers and date range.")
except Exception as e:
    print(f"Error downloading data: {e}")
    print("Please check your internet connection, ticker symbols, and date range.")
    exit() # Exit if data download fails

log_returns = np.log(data / data.shift(1)).dropna()

if log_returns.empty:
    raise ValueError("Not enough data to calculate log returns. Adjust date range.")

# --- Estimate Mean and Covariance ---
# Daily mean returns and covariance matrix
mu_daily = log_returns.mean().values
cov_daily = log_returns.cov().values

# --- Cholesky Decomposition for Correlation ---
# L is the lower triangular matrix such that L * L.T = cov_daily
L = cholesky(cov_daily, lower=True)

# --- Initialize Simulation Variables ---
initial_prices = data.iloc[-1].values
# Initialize array to store the final portfolio value for each simulation
final_portfolio_values = np.zeros(n_simulations)

# --- Monte Carlo Simulation ---
print(f"Running {n_simulations} simulations over {n_trading_days} days...")
# Pre-calculate common terms for efficiency
drift_daily = mu_daily - 0.5 * np.diag(cov_daily)

for i in range(n_simulations):
    # Generate uncorrelated random samples (standard normal)
    # Shape: (number_of_tickers, number_of_trading_days)
    uncorrelated_random_shocks = np.random.normal(size=(len(tickers), n_trading_days))

    # Apply Cholesky decomposition to introduce correlation
    correlated_random_shocks = np.dot(L, uncorrelated_random_shocks)

    # Initialize price paths for this simulation
    # Shape: (number_of_trading_days, number_of_tickers)
    price_paths = np.zeros((n_trading_days, len(tickers)))
    price_paths[0] = initial_prices

    # Simulate daily price movements
    for t in range(1, n_trading_days):
        # Geometric Brownian Motion formula for price at time t
        # P_t = P_{t-1} * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
        # Here dt = 1 (daily step), sigma*sqrt(dt)*Z is correlated_random_shocks[:, t-1]
        # (Note: Using t-1 for shocks to align with previous day's price)
        price_paths[t] = price_paths[t-1] * np.exp(drift_daily + correlated_random_shocks[:, t])

    # Aggregate weighted portfolio value at the end of the simulation
    # Portfolio value for this simulation at the last day
    final_portfolio_values[i] = np.dot(price_paths[-1], weights)

# --- Analyze Simulation Results ---
initial_portfolio_value = np.dot(initial_prices, weights)
returns = final_portfolio_values / initial_portfolio_value - 1

# Value at Risk (VaR) and Expected Shortfall (ES)
# VaR_95: The maximum loss at a 95% confidence level.
# ES_95: The average loss beyond the 95% VaR.
VaR_95 = np.percentile(returns, 5)
ES_95 = returns[returns <= VaR_95].mean()

# --- Plotting Results ---
plt.figure(figsize=(10, 6))
plt.hist(returns, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(VaR_95, color='red', linestyle='--', label=f'5% VaR: {VaR_95:.2%}')
plt.axvline(ES_95, color='purple', linestyle='--', label=f'ES: {ES_95:.2%}')
plt.title('Monte Carlo Simulation: Portfolio Return Distribution (1 Year Horizon)')
plt.xlabel('Portfolio Return')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Print Results ---
print("\n--- Simulation Results ---")
print(f"Initial Portfolio Value: ${initial_portfolio_value:,.2f}")
print(f"Simulations performed: {n_simulations}")
print(f"Simulation horizon: {n_trading_days} trading days")
print(f"Mean Simulated Return: {returns.mean():.2%}")
print(f"Standard Deviation of Returns: {returns.std():.2%}")
print(f"5% Value at Risk (VaR): {VaR_95:.2%}")
print(f"Expected Shortfall (ES): {ES_95:.2%}")