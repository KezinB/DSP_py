import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# RLS Algorithm Implementation
def rls(x, d, order, lambda_factor, delta):
    N = len(x)
    w = np.zeros((N, order))
    P = np.eye(order) / delta
    y = np.zeros(N)
    e = np.zeros(N)

    for n in range(order, N):
        x_n = x[n:n-order:-1]
        y[n] = np.dot(w[n], x_n)
        e[n] = d[n] - y[n]

        K = P @ x_n / (lambda_factor + x_n.T @ P @ x_n)
        w[n] = w[n-1] + K * e[n]
        P = (P - np.outer(K, x_n.T @ P)) / lambda_factor

        # Debugging output
        if n % 100 == 0:  # Print every 100 iterations
            print(f"RLS Iteration {n}: y[{n}] = {y[n]}, e[{n}] = {e[n]}, w[{n}] = {w[n]}")
            
    return y, e, w

# FT-RLS Algorithm Implementation
def ft_rls(x, d, order, lambda_factor, delta):
    N = len(x)
    w = np.zeros((N, order))
    f = np.zeros(order)
    b = np.zeros(order)
    g = np.zeros(order)
    y = np.zeros(N)
    e = np.zeros(N)
    rho = delta

    f[0] = 1.0
    b[0] = 1.0

    for n in range(order, N):
        x_n = x[n:n-order:-1]

        y[n] = np.dot(w[n], x_n)
        e[n] = d[n] - y[n]

        rho = lambda_factor * rho + np.dot(f, x_n) ** 2
        g = f / rho

        w[n] = w[n-1] + g * e[n]

        f = lambda_factor * (f - np.dot(x_n, g))
        b = lambda_factor * (b - np.dot(g, x_n))

        # Debugging output
        if n % 100 == 0:  # Print every 100 iterations
            print(f"FT-RLS Iteration {n}: y[{n}] = {y[n]}, e[{n}] = {e[n]}, w[{n}] = {w[n]}")
            
    return y, e, w

# Test Algorithms
np.random.seed(0)
N = 500
t = np.linspace(0, 1, N)
x = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, N)
d = np.sin(2 * np.pi * 5 * t)

order = 4
lambda_factor = 0.99
delta = 1e-2

y_rls, e_rls, w_rls = rls(x, d, order, lambda_factor, delta)
y_ftrls, e_ftrls, w_ftrls = ft_rls(x, d, order, lambda_factor, delta)

# Create folder with date, time, and code name
now = datetime.now()
folder_name = now.strftime("%Y%m%d_%H%M%S_RLS_VS_FT-RLS")
os.makedirs(folder_name, exist_ok=True)

# Save time series data as TXT and CSV
np.savetxt(os.path.join(folder_name, "data_rls.txt"), np.c_[t, y_rls, e_rls], header="Time\tRLS_Output\tRLS_Error", delimiter='\t')
np.savetxt(os.path.join(folder_name, "data_ftrls.txt"), np.c_[t, y_ftrls, e_ftrls], header="Time\tFTRLS_Output\tFTRLS_Error", delimiter='\t')

np.savetxt(os.path.join(folder_name, "data_rls.csv"), np.c_[t, y_rls, e_rls], header="Time,RLS_Output,RLS_Error", delimiter=',', comments='')
np.savetxt(os.path.join(folder_name, "data_ftrls.csv"), np.c_[t, y_ftrls, e_ftrls], header="Time,FTRLS_Output,FTRLS_Error", delimiter=',', comments='')

# Save weights separately
np.savetxt(os.path.join(folder_name, "weights_rls.txt"), w_rls, header="\t".join([f"Weights_{i}" for i in range(order)]), delimiter='\t')
np.savetxt(os.path.join(folder_name, "weights_ftrls.txt"), w_ftrls, header="\t".join([f"Weights_{i}" for i in range(order)]), delimiter='\t')

np.savetxt(os.path.join(folder_name, "weights_rls.csv"), w_rls, header=",".join([f"Weights_{i}" for i in range(order)]), delimiter=',', comments='')
np.savetxt(os.path.join(folder_name, "weights_ftrls.csv"), w_ftrls, header=",".join([f"Weights_{i}" for i in range(order)]), delimiter=',', comments='')

# Save the code itself to the same folder
with open(os.path.join(folder_name, "combined_code.py"), "w") as code_file:
    code_file.write(open(__file__).read())

# Plotting RLS
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(t, d, label="Desired Signal", color="green")
plt.title("Desired Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(t, x, label="Noisy Input", color="red", alpha=0.7)
plt.title("Noisy Input")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t, y_rls, label="RLS Filter Output", color="blue")
plt.title("RLS Filter Output")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

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
plt.savefig(os.path.join(folder_name, "RLS_Plots.png"))

# Plotting FT-RLS
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(t, d, label="Desired Signal", color="green")
plt.title("Desired Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(t, x, label="Noisy Input", color="red", alpha=0.7)
plt.title("Noisy Input")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t, y_ftrls, label="FT-RLS Filter Output", color="blue")
plt.title("FT-RLS Filter Output")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

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
plt.savefig(os.path.join(folder_name, "FT_RLS_Plots.png"))

plt.show()
