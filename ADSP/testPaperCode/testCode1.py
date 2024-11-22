import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# NOISE GENERATION
M = 25
N = 100
Delay = 3
lambda_ = 1

noise = np.random.randn(N)
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

# FILTERING ALGORITHM
for i in range(1, len(fnoise)):
    # Forward A Priori Prediction Error
    ep0 = np.random.randn() * ep0 - w0
    # Forward A Posteriori Prediction Error
    ep1 = ep0 / (gamma + 1)
    # MWLS Forward Error
    epMWL = np.random.randn() * ep1
    # Forward Weight Update
    phi += ep1 / (lambda_ * ep1 - 1) * (1 - w0)
    w0 += phi * ep1 / (lambda_ * ep1 - 1) * ep0
    # M+1 Conversion Factor
    gamma1 = gamma / (1 + gamma)
    # Backward A Priori Prediction Error
    ep0 = np.random.randn() * ep1 + phi
    # Backward A Posteriori Prediction Error
    ep1 = ep0 / (gamma + 1)
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

# Print the results
print("Filtering completed successfully!")
print("Final weights:", w0)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(fnoise, label='Filtered Noise')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Filtered Noise Signal')
plt.legend()
plt.grid(True)
plt.show()
