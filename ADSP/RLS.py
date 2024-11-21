import numpy as np
import matplotlib.pyplot as plt

# RLS Algorithm Implementation
def rls(x, d, order, lambda_factor, delta):
    N = len(x)
    w = np.zeros(order)  # Initialize weights
    P = np.eye(order) / delta  # Inverse correlation matrix
    y = np.zeros(N)  # Output of the filter
    e = np.zeros(N)  # Error signal

    for n in range(order, N):
        x_n = x[n:n-order:-1]  # Input vector
        y[n] = np.dot(w, x_n)  # Filter output
        e[n] = d[n] - y[n]  # Error signal

        # RLS Update Equations
        K = P @ x_n / (lambda_factor + x_n.T @ P @ x_n)
        w = w + K * e[n]
        P = (P - np.outer(K, x_n.T @ P)) / lambda_factor

    return y, e, w

# Test RLS Algorithm
np.random.seed(0)
N = 500
t = np.linspace(0, 1, N)
x = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, N)  # Noisy sinusoid
d = np.sin(2 * np.pi * 5 * t)  # Desired clean signal

order = 4
# lambda_factor = 0.99
lambda_factor = 0.99
delta = 1e-2

y_rls, e_rls, w_rls = rls(x, d, order, lambda_factor, delta)

# Plotting
plt.figure(figsize=(12, 10))

# Plot 1: Desired Signal
plt.subplot(4, 1, 1)
plt.plot(t, d, label="Desired Signal", color="green")
plt.title("Desired Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# Plot 2: Noisy Input
plt.subplot(4, 1, 2)
plt.plot(t, x, label="Noisy Input", color="red", alpha=0.7)
plt.title("Noisy Input")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# Plot 3: RLS Filter Output
plt.subplot(4, 1, 3)
plt.plot(t, y_rls, label="RLS Filter Output", color="blue")
plt.title("RLS Filter Output")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# Plot 4: Combined Signals
plt.subplot(4, 1, 4)
plt.plot(t, d, label="Desired Signal", color="green")
plt.plot(t, x, label="Noisy Input", color="red", alpha=0.7)
plt.plot(t, y_rls, label="RLS Filter Output", color="blue")
plt.legend()
plt.title("Combined Signals")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
