import numpy as np
import matplotlib.pyplot as plt

# RLS Algorithm
def rls(x, d, order, lambda_factor, delta):
    """
    Implements the Recursive Least Squares (RLS) algorithm.
    
    Parameters:
    - x: Input signal (noisy signal).
    - d: Desired signal (clean reference signal).
    - order: Filter order (number of coefficients).
    - lambda_factor: Forgetting factor (0 < lambda < 1).
    - delta: Regularization factor to initialize P matrix.

    Returns:
    - y: Filtered output.
    - e: Error signal (difference between desired and output).
    - w: Final weights of the adaptive filter.
    """
    N = len(x)  # Length of the input signal
    w = np.zeros(order)  # Initialize filter coefficients to zero
    P = np.eye(order) / delta  # Initialize the inverse correlation matrix
    y = np.zeros(N)  # Initialize output signal
    e = np.zeros(N)  # Initialize error signal

    # Main RLS loop
    for n in range(order, N):
        x_n = x[n:n-order:-1]  # Extract a slice of the input signal (most recent values)
        y[n] = np.dot(w, x_n)  # Compute the output signal as dot product of weights and input
        e[n] = d[n] - y[n]  # Compute the error signal
        
        # Compute the gain vector
        K = P @ x_n / (lambda_factor + x_n.T @ P @ x_n)
        
        # Update the filter coefficients
        w = w + K * e[n]
        
        # Update the inverse correlation matrix
        P = (P - np.outer(K, x_n.T @ P)) / lambda_factor

    return y, e, w

# FT-RLS Algorithm
def ft_rls(x, d, order, lambda_factor, delta):
    """
    Implements the Fast Transversal Recursive Least Squares (FT-RLS) algorithm.
    
    Parameters:
    - x: Input signal (noisy signal).
    - d: Desired signal (clean reference signal).
    - order: Filter order (number of coefficients).
    - lambda_factor: Forgetting factor (0 < lambda < 1).
    - delta: Regularization factor for initializing rho.

    Returns:
    - y: Filtered output.
    - e: Error signal (difference between desired and output).
    - w: Final weights of the adaptive filter.
    """
    N = len(x)  # Length of the input signal
    w = np.zeros(order)  # Initialize filter coefficients to zero
    f = np.zeros(order)  # Forward prediction coefficients
    b = np.zeros(order)  # Backward prediction coefficients
    g = np.zeros(order)  # Gain vector
    y = np.zeros(N)  # Initialize output signal
    e = np.zeros(N)  # Initialize error signal
    rho = delta  # Initialize a scalar normalization parameter

    f[0] = 1.0  # Initial value for forward prediction
    b[0] = 1.0  # Initial value for backward prediction

    # Main FT-RLS loop
    for n in range(order, N):
        x_n = x[n:n-order:-1]  # Extract a slice of the input signal (most recent values)

        y[n] = np.dot(w, x_n)  # Compute the output signal as dot product of weights and input
        e[n] = d[n] - y[n]  # Compute the error signal

        # Update rho (denominator of gain vector)
        rho = lambda_factor * rho + np.dot(f, x_n) ** 2
        
        g = f / rho  # Compute gain vector
        
        # Update filter coefficients
        w = w + g * e[n]

        # Update forward and backward prediction coefficients
        f = lambda_factor * (f - np.dot(x_n, g))
        b = lambda_factor * (b - np.dot(g, x_n))

    return y, e, w

# Generate test data
np.random.seed(0)  # Set seed for reproducibility
N = 500  # Number of samples
t = np.linspace(0, 1, N)  # Time vector
desired_signal = np.sin(2 * np.pi * 5 * t)  # Generate a clean sinusoidal signal
noise = np.random.normal(0, 0.5, N)  # Additive Gaussian noise
noisy_input = desired_signal + noise  # Combine clean signal with noise

# Parameters for both algorithms
order = 25  # Filter order (number of taps)
lambda_factor = 0.99  # Forgetting factor for RLS and FT-RLS
delta = 0.1  # Regularization parameter

# Apply RLS algorithm
y_rls, e_rls, w_rls = rls(noisy_input, desired_signal, order, lambda_factor, delta)

# Apply FT-RLS algorithm
y_ftrls, e_ftrls, w_ftrls = ft_rls(noisy_input, desired_signal, order, lambda_factor, delta)

# Plot Results
plt.figure(figsize=(12, 10))

# Desired Signal
plt.subplot(4, 1, 1)
plt.plot(t, desired_signal, label="Desired Signal", color="green")
plt.title("Desired Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

# RLS Output
plt.subplot(4, 1, 2)
plt.plot(t, noisy_input, label="Noisy Input", color="red", alpha=0.7)
plt.plot(t, y_rls, label="RLS Output", color="blue")
plt.title("RLS Algorithm Output")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

# FT-RLS Output
plt.subplot(4, 1, 3)
plt.plot(t, noisy_input, label="Noisy Input", color="red", alpha=0.7)
plt.plot(t, y_ftrls, label="FT-RLS Output", color="purple")
plt.title("FT-RLS Algorithm Output")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

# Combined Comparison
plt.subplot(4, 1, 4)
plt.plot(t, desired_signal, label="Desired Signal", color="green")
plt.plot(t, y_rls, label="RLS Output", color="blue")
plt.plot(t, y_ftrls, label="FT-RLS Output", color="purple")
plt.title("Comparison of RLS and FT-RLS")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
