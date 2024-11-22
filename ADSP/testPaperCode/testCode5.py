import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# NOISE GENERATION
M = 25
N = 1000
lambda_ = 0.99

t = np.arange(N)
sine_wave = np.sin(2 * np.pi * 0.05 * t)
noise = sine_wave + np.random.randn(N) * 0.5  # Noised sine wave
num, den = butter(10, 0.5)
fnoise = lfilter(num, den, noise)
fnoise = fnoise / np.std(fnoise)

# RLS Algorithm Implementation
def rls_filter(noise, M, lambda_):
    N = len(noise)
    P = np.eye(M) / 0.01
    w = np.zeros(M)
    e = np.zeros(N)
    u = np.zeros((N, M))
    msd = np.zeros(N)
    
    for i in range(M, N):
        u[i, :] = noise[i-M:i]
        k = P @ u[i, :] / (lambda_ + u[i, :] @ P @ u[i, :])
        e[i] = noise[i] - w @ u[i, :]
        w += k * e[i]
        P = (P - np.outer(k, u[i, :]) @ P) / lambda_
        msd[i] = np.mean((w - w) ** 2)  # Placeholder for actual MSD calculation
    
    return e, w, msd

# FT-RLS Algorithm Implementation
def ft_rls_filter(noise, M, lambda_):
    N = len(noise)
    forward_weights = np.zeros((N, M))
    backward_weights = np.zeros((N, M))
    forward_error = np.zeros(N)
    backward_error = np.zeros(N)
    u = np.zeros((N, M))
    msd_forward = np.zeros(N)
    msd_backward = np.zeros(N)
    
    P_forward = np.eye(M) / 0.01
    P_backward = np.eye(M) / 0.01
    
    # Forward filtering
    for i in range(M, N):
        u[i, :] = noise[i-M:i]
        k = P_forward @ u[i, :] / (lambda_ + u[i, :] @ P_forward @ u[i, :])
        forward_error[i] = noise[i] - forward_weights[i-1] @ u[i, :]
        forward_weights[i] = forward_weights[i-1] + k * forward_error[i]
        P_forward = (P_forward - np.outer(k, u[i, :]) @ P_forward) / lambda_
        msd_forward[i] = np.mean((forward_weights[i] - forward_weights[i]) ** 2)  # Placeholder for actual MSD calculation
    
    # Backward filtering
    for i in range(N-M-1, -1, -1):
        u[i, :] = noise[i+1:i+M+1]
        k = P_backward @ u[i, :] / (lambda_ + u[i, :] @ P_backward @ u[i, :])
        backward_error[i] = noise[i] - backward_weights[i+1] @ u[i, :]
        backward_weights[i] = backward_weights[i+1] + k * backward_error[i]
        P_backward = (P_backward - np.outer(k, u[i, :]) @ P_backward) / lambda_
        msd_backward[i] = np.mean((backward_weights[i] - backward_weights[i]) ** 2)  # Placeholder for actual MSD calculation
    
    smoothed_signal = (forward_weights + backward_weights) / 2
    msd = np.mean((forward_error + backward_error) ** 2)
    return smoothed_signal, forward_error, backward_error, msd

# Perform RLS Filtering
rls_output, rls_weights, msd_rls = rls_filter(fnoise, M, lambda_)

# Perform FT-RLS Filtering
ft_rls_output, forward_error, backward_error, msd_ft_rls = ft_rls_filter(fnoise, M, lambda_)

# Print the results
print("RLS Filtering completed successfully!")
print("FT-RLS Filtering completed successfully!")

# Plotting the results
plt.figure(figsize=(14, 18))

plt.subplot(5, 1, 1)
plt.plot(t, sine_wave, label='Original Sine Wave')
plt.plot(t, noise, label='Noised Sine Wave', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Input Sine Wave and Noised Sine Wave')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(fnoise, label='Filtered Noise')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Filtered Noise Signal')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(t, rls_output, label='RLS Output')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('RLS Filtered Output')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t, ft_rls_output[:, -1], label='FT-RLS Output')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('FT-RLS Filtered Output')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(msd_rls, label='RLS Mean Squared Deviation (MSD)')
plt.plot(msd_ft_rls, label='FT-RLS Mean Squared Deviation (MSD)')
plt.xlabel('Iteration')
plt.ylabel('MSD')
plt.title('Mean Squared Deviation over Iterations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
