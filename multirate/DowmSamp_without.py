import numpy as np
import matplotlib.pyplot as plt

# Define the sampling frequency and time vector
fs = 1000
t = np.arange(0, 1, 1/fs)

# Create an input signal without high-frequency components (no aliasing)
input_signal = np.sin(2 * np.pi * 100 * t)

# Define the downsampling factor
downsampling_factor = 3

# Perform down-sampling
output_signal = input_signal[::downsampling_factor]

# Plot input and downsampled signals in the time domain
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, input_signal)
plt.title("Input Signal (Without Aliasing)")
plt.xlabel("Time (s)")
plt.grid()

t_downsampled = np.arange(0, len(output_signal)) * (1 / (fs / downsampling_factor))
plt.subplot(2, 1, 2)
plt.plot(t_downsampled, output_signal)
plt.title("Output Signal (Down-sampled)")
plt.xlabel("Time (s)")
plt.grid()

plt.tight_layout()
plt.show()

# Plot the frequency spectrum for a factor of 3 down-sampler without aliasing
plt.figure(figsize=(8, 4))
frequencies = np.fft.fftfreq(len(output_signal), 1 / fs)
spectrum = np.fft.fft(output_signal)
plt.plot(frequencies, np.abs(spectrum))
plt.title("Frequency Spectrum (Without Aliasing)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()
