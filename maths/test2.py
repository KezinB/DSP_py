import numpy as np
import matplotlib.pyplot as plt

# Parameters for Jump Diffusion Model
S0 = 100  # Initial stock price
mu = 0.05  # Drift (mean return)
sigma = 0.2  # Volatility (standard deviation of returns)
lambda_jump = 0.1  # Poisson jump intensity (on average 10% jumps per year)
mu_jump = 0.1  # Mean size of jumps (10% jump size)
sigma_jump = 0.05  # Standard deviation of jump sizes (5% jump size)
T = 1  # Time horizon (1 year)
dt = 1/252  # Time step (daily data)
N = int(T / dt)  # Number of time steps
M = 10  # Number of simulations

# Simulate stock price paths with jumps
np.random.seed(42)  # For reproducibility
time = np.linspace(0, T, N)
simulated_prices = np.zeros((M, N))
for i in range(M):
    # Generate random normal noise for the GBM component
    dW = np.random.normal(0, np.sqrt(dt), N)
    # Generate jumps from the Poisson distribution
    jump_times = np.random.poisson(lambda_jump * dt, N)
    jump_sizes = np.random.normal(mu_jump, sigma_jump, N)
    
    # Simulate stock price path with jumps
    S = np.zeros(N)
    S[0] = S0
    for t in range(1, N):
        # GBM part
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])
        # Jump part (apply jumps at random times)
        if jump_times[t] > 0:
            S[t] *= np.exp(jump_sizes[t])  # Apply jump to stock price

    simulated_prices[i, :] = S

# Plot the simulated stock price paths with jumps
plt.figure(figsize=(10, 6))
for i in range(M):
    plt.plot(time, simulated_prices[i, :], label=f'Simulation {i+1}')
plt.title("Stock Price Simulation with Jumps (Jump Diffusion Model)")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.legend()
plt.show()
