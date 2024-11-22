import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# NOISE GENERATION
M = 25
N = 1000
Delay = 3
lambda_ = 1

t = np.arange(N)
sine_wave = np.sin(2 * np.pi * 0.05 * t)
noise = sine_wave + np.random.randn(N) * 0.5  # Noised sine wave
num, den = butter(10, 0.5)
fnoise = lfilter(num, den, noise)
fnoise = fnoise / np.std(fnoise)
weights = np.zeros((M, len(fnoise) + 1))
signal = fnoise

# INITIALIZATION
epsilon = 0.0001
u = np.zeros((N, M))
e = np.zeros(N)
w0 = np.zeros(M)
w1 = np.zeros(M)
phi = np.zeros(M)
gamma = np.array([1])
ep0 = np.ones(M) * epsilon
ep1 = np.ones(M) * epsilon

# Intermediate signals
forward_apriori_error = []
forward_aposteriori_error = []
backward_apriori_error = []
backward_aposteriori_error = []
msd = []

# FILTERING ALGORITHM
for i in range(1, len(fnoise)):
    # Forward A Priori Prediction Error
    ep0 = np.random.randn() * ep0 - w0
    forward_apriori_error.append(ep0.copy())
    # Forward A Posteriori Prediction Error
    ep1 = ep0 / (gamma + 1)
    forward_aposteriori_error.append(ep1.copy())
    # MWLS Forward Error
    epMWL = np.random.randn() * ep1
    # Forward Weight Update
    phi += ep1 / (lambda_ * ep1 - 1) * (1 - w0)
    w0 += phi * ep1 / (lambda_ * ep1 - 1) * ep0
    # M+1 Conversion Factor
    gamma1 = gamma / (1 + gamma)
    # Backward A Priori Prediction Error
    ep0 = np.random.randn() * ep1 + phi
    backward_apriori_error.append(ep0.copy())
    # Backward A Posteriori Prediction Error
    ep1 = ep0 / (gamma + 1)
    backward_aposteriori_error.append(ep1.copy())
    # MWLS Backward Error
    epMWL = np.random.randn() * ep1
    # Backward Weight Update
    phi += ep1 / (lambda_ * ep1 - 1) * (1 - w0)
    w0 += phi * ep1 / (lambda_ * ep1 - 1) * ep0
    # M+1 Conversion Factor
    gamma1 = gamma / (1 + gamma)
    # New Sample of Input Data
    u[i, :] = u[i - 1, :] + ep1 / (lambda_ * ep1 - 1) * (1 - w0)
    # Joint-Estimation Error
    epJ = ep1 / (gamma + 1)
    # A Posteriori Joint-Estimation Error
    epJ = epJ / (gamma + 1)
    # Joint-Estimation Weight Update
    w0 += phi * ep1 / (lambda_ * ep1 - 1) * ep0

    # Calculate MSD
    msd.append(np.mean((w0 - w1) ** 2))

# Print the results
print("Filtering completed successfully!")
print("Final weights:", w0)

# Plotting the results
plt.figure(figsize=(14, 12))

plt.subplot(4, 1, 1)
plt.plot(t, sine_wave, label='Original Sine Wave')
plt.plot(t, noise, label='Noised Sine Wave', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Input Sine Wave and Noised Sine Wave')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(fnoise, label='Filtered Noise')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Filtered Noise Signal')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(forward_apriori_error, label='Forward A Priori Error', alpha=0.7)
plt.plot(forward_aposteriori_error, label='Forward A Posteriori Error', alpha=0.7)
plt.plot(backward_apriori_error, label='Backward A Priori Error', alpha=0.7)
plt.plot(backward_aposteriori_error, label='Backward A Posteriori Error', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.title('Intermediate Errors')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(msd, label='Mean Squared Deviation (MSD)')
plt.xlabel('Iteration')
plt.ylabel('MSD')
plt.title('Mean Squared Deviation over Iterations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
