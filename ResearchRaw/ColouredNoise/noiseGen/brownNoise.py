import numpy as np
import matplotlib.pyplot as plt

def brown_noise(N):
    white_noise = np.random.randn(N)
    brown_noise = np.cumsum(white_noise)
    return brown_noise / np.max(np.abs(brown_noise))  # Normalize

N = 1000  # Number of samples
noise = brown_noise(N)

# Plot the brown noise
plt.plot(noise)
plt.title('Brown Noise')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.show()
