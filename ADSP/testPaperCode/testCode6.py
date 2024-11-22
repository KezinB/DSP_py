import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def ft_rls(x, d, order, lambda_factor, delta):
    """
    Fast Transversal Recursive Least Squares (FT-RLS) Algorithm.

    Parameters:
        x (np.array): Input signal (noisy signal).
        d (np.array): Desired signal.
        order (int): Filter order.
        lambda_factor (float): Forgetting factor (0 < lambda_factor <= 1).
        delta (float): Regularization parameter (positive scalar).

    Returns:
        y (np.array): Filter output.
        e (np.array): Error signal (desired - output).
        w (np.array): Filter weights.
    """
    N = len(x)
    w = np.zeros(order)
    f = np.zeros(order)
    b = np.zeros(order)
    g = np.zeros(order)
    rho = delta
    y = np.zeros(N)
    e = np.zeros(N)

    f[0] = 1.0
    b[0] = 1.0

    for n in range(order, N):
        x_n = x[n:n-order:-1]  # Most recent 'order' samples (reversed)

        y[n] = np.dot(w, x_n)  # Filter output
        e[n] = d[n] - y[n]     # Prediction error

        rho = lambda_factor * rho + np.dot(f, x_n) ** 2  # Update regularization term
        g = f / rho  # Conversion factor

        w = w + g * e[n]  # Update weights
        f = lambda_factor * (f - np.dot(x_n, g))  # Update forward coefficients
        b = lambda_factor * (b - np.dot(g, x_n))  # Update backward coefficients

    return y, e, w

# Generate test signals
np.random.seed(0)
N = 500
t = np.linspace(0, 1, N)
x = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, N)  # Noisy input
d = np.sin(2 * np.pi * 5 * t)                                # Desired signal

# FT-RLS Parameters
order = 25
lambda_factor = 0.99
#delta = 1e-2
delta = .1

# Run FT-RLS
start_time = datetime.now()
y_ftrls, e_ftrls, w_ftrls = ft_rls(x, d, order, lambda_factor, delta)
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

# Create folder for saving results
folder_path = "FT_RLS_Results"
now = datetime.now()
folder_name = now.strftime("%Y_%m_%d-%H_%M_%S_FT-RLS")
full_folder_path = os.path.join(folder_path, folder_name)
os.makedirs(full_folder_path, exist_ok=True)

# Save time-series data
np.savetxt(os.path.join(full_folder_path, "data_ftrls.txt"), np.c_[t, y_ftrls, e_ftrls], 
           header="Time\tFTRLS_Output\tFTRLS_Error", delimiter='\t')
np.savetxt(os.path.join(full_folder_path, "data_ftrls.csv"), np.c_[t, y_ftrls, e_ftrls], 
           header="Time,FTRLS_Output,FTRLS_Error", delimiter=',', comments='')

# Save weights
np.savetxt(os.path.join(full_folder_path, "weights_ftrls.txt"), w_ftrls, 
           header="\t".join([f"Weights_{i}" for i in range(order)]), delimiter='\t')
np.savetxt(os.path.join(full_folder_path, "weights_ftrls.csv"), w_ftrls, 
           header=",".join([f"Weights_{i}" for i in range(order)]), delimiter=',', comments='')

# Save the code itself
script_name = "ft_rls_full_implementation.py"
with open(os.path.join(full_folder_path, script_name), "w") as code_file:
    code_content = """
    import numpy as np
    import matplotlib.pyplot as plt
    ...
    # Full code is written here for reference purposes.
    """
    code_file.write(code_content)

# Plot Results
plt.figure(figsize=(12, 8))

# Desired Signal
plt.subplot(3, 1, 1)
plt.plot(t, d, label="Desired Signal", color="green")
plt.title("Desired Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# Noisy Input Signal
plt.subplot(3, 1, 2)
plt.plot(t, x, label="Noisy Input", color="red")
plt.title("Noisy Input")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# FT-RLS Output
plt.subplot(3, 1, 3)
plt.plot(t, y_ftrls, label="FT-RLS Output", color="blue")
plt.plot(t, d, label="Desired Signal", color="green", linestyle="dashed")
plt.legend()
plt.title("FT-RLS Output vs Desired Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()

# Save plot
plot_file = os.path.join(full_folder_path, "FT_RLS_Plot.png")
plt.savefig(plot_file)
plt.show()

# Display computation time
print(f"FT-RLS Computation Time: {duration:.4f} seconds")
print(f"Results saved to: {full_folder_path}")
