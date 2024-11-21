import numpy as np
import matplotlib.pyplot as plt

# Create an input signal (e.g., a sine wave)
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1 / fs)  # Time vector from 0 to 1 second
input_signal = np.sin(2 * np.pi * 50 * t)  # A 50 Hz sine wave as the input signal

# Upsampling factor
upsampling_factor = 5

# Perform upsampling by inserting zeros between samples
output_signal = np.zeros(len(input_signal) * upsampling_factor)
output_signal[::upsampling_factor] = input_signal

# Time domain plot
plt.figure(figsize=(12, 6))

# Plot the input signal
plt.subplot(2, 1, 1)
plt.plot(t, input_signal, label="Input Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Input Signal")
plt.grid()
plt.legend()

# Plot the upsampled signal
t_upsampled = np.arange(0, 1, 1 / (fs * upsampling_factor))
plt.subplot(2, 1, 2)
plt.plot(t_upsampled, output_signal, label="Upsampled Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Upsampled Signal (Factor of 5)")
plt.grid()
plt.legend()

plt.tight_layout()

# Frequency domain plot
plt.figure(figsize=(12, 6))

# Plot the input spectrum
plt.subplot(2, 1, 1)
frequencies_input = np.fft.fftfreq(len(input_signal), 1 / fs)
spectrum_input = np.fft.fft(input_signal)
plt.plot(frequencies_input, np.abs(spectrum_input), label="Input Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Input Spectrum")
plt.grid()
plt.legend()

# Plot the upsampled spectrum
plt.subplot(2, 1, 2)
frequencies_upsampled = np.fft.fftfreq(len(output_signal), 1 / (fs * upsampling_factor))
spectrum_upsampled = np.fft.fft(output_signal)
plt.plot(frequencies_upsampled, np.abs(spectrum_upsampled), label="Upsampled Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Upsampled Spectrum (Factor of 5)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
