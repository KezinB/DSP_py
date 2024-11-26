import numpy as np
import matplotlib.pyplot as plt

def pink_noise(N):
    # Generate white noise
    white_noise = np.random.randn(N)
    # Apply a filter to shape the noise to pink noise
    pink_noise = np.cumsum(white_noise)
    return pink_noise

N = 1000  # Number of samples
noise = pink_noise(N)

# Plot the pink noise
plt.plot(noise)
plt.title('Pink Noise')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.show()
