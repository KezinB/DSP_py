import numpy as np
import matplotlib.pyplot as plt

# Parameters for GBM
S0 = 100  # Initial stock price
mu = 0.05  # Drift (mean return)
sigma = 0.2  # Volatility (standard deviation of returns)
T = 1  # Time horizon (1 year)
dt = 1/252  # Time step (daily data, 252 trading days in a year)
N = int(T / dt)  # Number of time steps
M = 10  # Number of simulations

# Simulate stock price paths
np.random.seed(42)  # For reproducibility
time = np.linspace(0, T, N)
simulated_prices = np.zeros((M, N))
for i in range(M):
    # Generate random normal noise for the Brownian motion
    dW = np.random.normal(0, np.sqrt(dt), N)  # Brownian increments
    # Simulate stock price path using the GBM model
    S = np.zeros(N)
    S[0] = S0
    for t in range(1, N):
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])
    simulated_prices[i, :] = S

# Plot the simulated stock price paths
plt.figure(figsize=(10, 6))
for i in range(M):
    plt.plot(time, simulated_prices[i, :], label=f'Simulation {i+1}')
plt.title("Stock Price Simulation Using Geometric Brownian Motion (GBM)")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.legend()
plt.show()
