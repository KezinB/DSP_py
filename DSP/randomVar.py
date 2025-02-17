import numpy as np
import matplotlib.pyplot as plt

# Function to generate random variables for different distributions
def generate_random_variables():
    # (a) Bernoulli Distribution: Outcome is 0 or 1 with probability b
    def bernoulli(b, size=1000):
        return np.random.binomial(1, b, size)

    # (b) Binomial Distribution: n trials, b probability of success
    def binomial(n, b, size=1000):
        return np.random.binomial(n, b, size)

    # (c) Geometric Distribution: Probability of success in each trial
    def geometric(b, size=1000):
        return np.random.geometric(b, size)

    # (d) Poisson Distribution: Rate (lambda) of occurrences
    def poisson(lam, size=1000):
        return np.random.poisson(lam, size)

    # (e) Uniform Distribution: Random number between low and high values
    def uniform(low, high, size=1000):
        return np.random.uniform(low, high, size)

    # (f) Gaussian (Normal) Distribution: Mean (mu), Standard Deviation (sigma)
    def gaussian(mu, sigma, size=1000):
        return np.random.normal(mu, sigma, size)

    # (g) Exponential Distribution: Rate parameter (lambda)
    def exponential(lam, size=1000):
        return np.random.exponential(1/lam, size)

    # (h) Laplacian Distribution: Mean (mu), Scale parameter (b)
    def laplacian(mu, b, size=1000):
        return np.random.laplace(mu, b, size)

    # Example parameters for each distribution
    bernoulli_b = 0.5
    binomial_n, binomial_b = 10, 0.5
    geometric_b = 0.5
    poisson_lambda = 5
    uniform_low, uniform_high = 0, 10
    gaussian_mu, gaussian_sigma = 0, 1
    exponential_lambda = 1
    laplacian_mu, laplacian_b = 0, 1

    # Generate random variables for each distribution
    bernoulli_data = bernoulli(bernoulli_b)
    binomial_data = binomial(binomial_n, binomial_b)
    geometric_data = geometric(geometric_b)
    poisson_data = poisson(poisson_lambda)
    uniform_data = uniform(uniform_low, uniform_high)
    gaussian_data = gaussian(gaussian_mu, gaussian_sigma)
    exponential_data = exponential(exponential_lambda)
    laplacian_data = laplacian(laplacian_mu, laplacian_b)

    # Plotting the distributions
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 3, 1)
    plt.hist(bernoulli_data, bins=2, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Bernoulli Distribution')

    plt.subplot(3, 3, 2)
    plt.hist(binomial_data, bins=range(binomial_n+1), alpha=0.7, color='green', edgecolor='black')
    plt.title('Binomial Distribution')

    plt.subplot(3, 3, 3)
    plt.hist(geometric_data, bins=range(1, max(geometric_data)+1), alpha=0.7, color='red', edgecolor='black')
    plt.title('Geometric Distribution')

    plt.subplot(3, 3, 4)
    plt.hist(poisson_data, bins=range(max(poisson_data)+1), alpha=0.7, color='purple', edgecolor='black')
    plt.title('Poisson Distribution')

    plt.subplot(3, 3, 5)
    plt.hist(uniform_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Uniform Distribution')

    plt.subplot(3, 3, 6)
    plt.hist(gaussian_data, bins=30, alpha=0.7, color='pink', edgecolor='black')
    plt.title('Gaussian Distribution')

    plt.subplot(3, 3, 7)
    plt.hist(exponential_data, bins=30, alpha=0.7, color='brown', edgecolor='black')
    plt.title('Exponential Distribution')

    plt.subplot(3, 3, 8)
    plt.hist(laplacian_data, bins=30, alpha=0.7, color='gray', edgecolor='black')
    plt.title('Laplacian Distribution')

    plt.tight_layout()
    plt.show()

# Call the function to generate and visualize random variables for the distributions
generate_random_variables()
