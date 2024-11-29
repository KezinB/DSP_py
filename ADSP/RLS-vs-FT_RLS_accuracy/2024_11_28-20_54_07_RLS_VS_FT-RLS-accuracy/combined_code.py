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


np.random.seed(0)
N = 1000
t = np.linspace(0, 1, N)
frequency = 15 # Specify the frequency you want
x = np.sin(2 * np.pi * frequency * t) + np.random.normal(0, 0.5, N)
d = np.sin(2 * np.pi * frequency * t)


order = 25
lambda_factor = 1
delta = 0.250
#delta = 0.01

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

# Create folder with date and time separated by an underscore and a dash
folder_path = "C:\\Users\\HP\\OneDrive\\Documents\\Codes\\python\\ADSP\\RLS-vs-FT_RLS_accuracy"
now = datetime.now()
folder_name = now.strftime("%Y_%m_%d-%H_%M_%S_RLS_VS_FT-RLS-accuracy")
full_folder_path = os.path.join(folder_path, folder_name)
os.makedirs(full_folder_path, exist_ok=True)

# Save time series data as TXT and CSV
np.savetxt(os.path.join(full_folder_path, "data_rls.txt"), np.c_[t, y_rls, e_rls], header="Time\tRLS_Output\tRLS_Error", delimiter='\t')
np.savetxt(os.path.join(full_folder_path, "data_ftrls.txt"), np.c_[t, y_ftrls, e_ftrls], header="Time\tFTRLS_Output\tFTRLS_Error", delimiter='\t')

np.savetxt(os.path.join(full_folder_path, "data_rls.csv"), np.c_[t, y_rls, e_rls], header="Time,RLS_Output,RLS_Error", delimiter=',', comments='')
np.savetxt(os.path.join(full_folder_path, "data_ftrls.csv"), np.c_[t, y_ftrls, e_ftrls], header="Time,FTRLS_Output,FTRLS_Error", delimiter=',', comments='')

# Save weights separately
np.savetxt(os.path.join(full_folder_path, "weights_rls.txt"), w_rls, header="\t".join([f"Weights_{i}" for i in range(order)]), delimiter='\t')
np.savetxt(os.path.join(full_folder_path, "weights_ftrls.txt"), w_ftrls, header="\t".join([f"Weights_{i}" for i in range(order)]), delimiter='\t')

np.savetxt(os.path.join(full_folder_path, "weights_rls.csv"), w_rls, header=",".join([f"Weights_{i}" for i in range(order)]), delimiter=',', comments='')
np.savetxt(os.path.join(full_folder_path, "weights_ftrls.csv"), w_ftrls, header=",".join([f"Weights_{i}" for i in range(order)]), delimiter=',', comments='')

# Save the code itself to the same folder
with open(os.path.join(full_folder_path, "combined_code.py"), "w") as code_file:
    code_file.write(open(__file__).read())

# Plotting RLS with timing
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

plt.figtext(0.5, 0.01, f"RLS: Start Time: {start_time_rls}, End Time: {end_time_rls}, Duration: {duration_rls:.4f} seconds", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig(os.path.join(full_folder_path, "RLS_Plots.png"))

# Plotting FT-RLS with timing
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

# Calculate MSE for both RLS and FT-RLS
mse_rls = np.mean(e_rls ** 2)
mse_ftrls = np.mean(e_ftrls ** 2)

# Create trade-off plot
plt.figure(figsize=(8, 6))
plt.scatter(duration_rls, mse_rls, color='blue', label='RLS', s=100, alpha=0.7)
plt.scatter(duration_ftrls, mse_ftrls, color='orange', label='FT-RLS', s=100, alpha=0.7)

# Annotate points with their labels
plt.text(duration_rls, mse_rls, "RLS", fontsize=10, ha='right', color='blue')
plt.text(duration_ftrls, mse_ftrls, "FT-RLS", fontsize=10, ha='right', color='orange')

# Add labels, title, and grid
plt.title("Trade-off Between Execution Time and Accuracy", fontsize=14)
plt.xlabel("Execution Time (seconds)", fontsize=12)
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save and show plot
plt.savefig(os.path.join(full_folder_path, "Tradeoff_Plot.png"))
plt.show()

plt.figtext(0.5, 0.01, f"FT-RLS: Start Time: {start_time_ftrls}, End Time: {end_time_ftrls}, Duration: {duration_ftrls:.4f} seconds", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig(os.path.join(full_folder_path, "FT_RLS_Plots.png"))

plt.show()
