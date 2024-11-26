import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def blue_noise(N):
    white_noise = np.random.randn(N)
    b = [0.5, -0.5]
    a = [1, -0.98]
    blue_noise = lfilter(b, a, white_noise)
    return blue_noise

N = 1000  # Number of samples
noise = blue_noise(N)

# Plot the blue noise
plt.plot(noise)
plt.title('Blue Noise')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.show()
