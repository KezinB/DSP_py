import numpy as np
import matplotlib.pyplot as plt

# FT-RLS Algorithm Implementation
def ft_rls(x, d, order, lambda_factor, delta):
    N = len(x)
    w = np.zeros(order)  # Initialize weights
    f = np.zeros(order)  # Forward error
    b = np.zeros(order)  # Backward error
    g = np.zeros(order)  # Gain vector
    y = np.zeros(N)  # Filter output
    e = np.zeros(N)  # Error signal
    rho = delta  # Initialize forgetting factor weight

    # Initialize forward and backward error vectors
    f[0] = 1.0
    b[0] = 1.0

    for n in range(order, N):
        x_n = x[n:n-order:-1]  # Input vector

        # Compute filter output and forward error
        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]

        # Update gain vector
        rho = lambda_factor * rho + np.dot(f, x_n) ** 2
        g = f / rho

        # Update weights
        w = w + g * e[n]

        # Update forward and backward error
        f = lambda_factor * (f - np.dot(x_n, g))
        b = lambda_factor * (b - np.dot(g, x_n))

    return y, e, w

# Test FT-RLS Algorithm
np.random.seed(0)
N = 500
t = np.linspace(0, 1, N)
x = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, N)  # Noisy sinusoid
d = np.sin(2 * np.pi * 5 * t)  # Desired clean signal

order = 4
lambda_factor = 0.99
delta = 1e-2

y_ftrls, e_ftrls, w_ftrls = ft_rls(x, d, order, lambda_factor, delta)

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

# Plot 3: FT-RLS Filter Output
plt.subplot(4, 1, 3)
plt.plot(t, y_ftrls, label="FT-RLS Filter Output", color="blue")
plt.title("FT-RLS Filter Output")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# Plot 4: Combined Signals
plt.subplot(4, 1, 4)
plt.plot(t, d, label="Desired Signal", color="green")
plt.plot(t, x, label="Noisy Input", color="red", alpha=0.7)
plt.plot(t, y_ftrls, label="FT-RLS Filter Output", color="blue")
plt.legend()
plt.title("Combined Signals")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
