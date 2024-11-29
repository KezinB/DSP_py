import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# RLS Algorithm Implementation
def rls(x, d, order, lambda_factor, delta):
    N = len(x)
    w = np.zeros(order)
    P = np.eye(order) / delta
    y = np.zeros(N)
    e = np.zeros(N)

    for n in range(order, N):
        x_n = x[n:n-order:-1]
        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]

        K = P @ x_n / (lambda_factor + x_n.T @ P @ x_n)
        w = w + K * e[n]
        P = (P - np.outer(K, x_n.T @ P)) / lambda_factor

    return y, e, w

# FT-RLS Algorithm Implementation
def ft_rls(x, d, order, lambda_factor, delta):
    N = len(x)
    w = np.zeros(order)
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

        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]

        rho = lambda_factor * rho + np.dot(f, x_n) ** 2
        g = f / rho

        w = w + g * e[n]

        f = lambda_factor * (f - np.dot(x_n, g))
        b = lambda_factor * (b - np.dot(g, x_n))

    return y, e, w

# Function Definitions (RLS and FT-RLS) remain the same

np.random.seed(0)
N = 1000
t = np.linspace(0, 1, N)

# Testing with multiple frequencies
frequencies = [10, 20]  # Adjust frequency range here
lambda_factor = 1
delta = 0.250
# lambda_factor = 0.99  # Updated lambda_factor
# delta = 0.1  # Updated delta

results_folder = "C:\\Users\\user\\OneDrive\\Documents\\Codes\\python\\ADSP\\RLS-VS-FT_RLS_Parameter_Update"
now = datetime.now()
folder_name = now.strftime("%Y_%m_%d-%H_%M_%S_Parameter_Test")
full_folder_path = os.path.join(results_folder, folder_name)
os.makedirs(full_folder_path, exist_ok=True)

# Save the code itself to the same folder
with open(os.path.join(full_folder_path, "combined_code.py"), "w") as code_file:
    code_file.write(open(__file__).read())

for frequency in frequencies:
    x = np.sin(2 * np.pi * frequency * t) + np.random.normal(0, 0.5, N)
    d = np.sin(2 * np.pi * frequency * t)

    order = 25

    # Measure time for RLS
    start_time_rls = datetime.now()
    y_rls, e_rls, w_rls = rls(x, d, order, lambda_factor, delta)
    end_time_rls = datetime.now()
    duration_rls = (end_time_rls - start_time_rls).total_seconds()

    # Measure time for FT-RLS
    start_time_ftrls = datetime.now()
    y_ftrls, e_ftrls, w_ftrls = ft_rls(x, d, order, lambda_factor, delta)
    end_time_ftrls = datetime.now()
    duration_ftrls = (end_time_ftrls - start_time_ftrls).total_seconds()

    # Plotting for each frequency
    plt.figure(figsize=(12, 8))
    plt.plot(t, d, label="Desired Signal", color="green")
    plt.plot(t, x, label="Noisy Input", color="red", alpha=0.7)
    plt.plot(t, y_rls, label="RLS Output", color="blue")
    plt.plot(t, y_ftrls, label="FT-RLS Output", color="orange")
    plt.title(f"RLS vs FT-RLS at Frequency {frequency} Hz")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.figtext(0.5, 0.01, f"RLS Duration: {duration_rls:.4f} seconds | FT-RLS Duration: {duration_ftrls:.4f} seconds", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
    
    # Save plot
    plt.tight_layout()
    plot_name = f"Comparison_Frequency_{frequency}_Hz.png"
    plt.savefig(os.path.join(full_folder_path, plot_name))
    plt.close()

    # Save Results
    np.savetxt(os.path.join(full_folder_path, f"RLS_Frequency_{frequency}_Hz.csv"), np.c_[t, y_rls, e_rls], header="Time,RLS_Output,RLS_Error", delimiter=',', comments='')
    np.savetxt(os.path.join(full_folder_path, f"FT_RLS_Frequency_{frequency}_Hz.csv"), np.c_[t, y_ftrls, e_ftrls], header="Time,FT_RLS_Output,FT_RLS_Error", delimiter=',', comments='')
