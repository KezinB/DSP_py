import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def violet_noise(N):
    white_noise = np.random.randn(N)
    b = [1, -1]
    a = [1]
    violet_noise = lfilter(b, a, white_noise)
    return violet_noise

N = 1000  # Number of samples
noise = violet_noise(N)

# Plot the violet noise
plt.plot(noise)
plt.title('Violet Noise')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.show()
