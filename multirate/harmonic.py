import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000  # Sampling frequency (Hz)
T = 1      # Duration (s)
t = np.linspace(0, T, int(fs * T), endpoint=False)  # Time vector

# Fundamental frequency and harmonics
f1 = 50  # Fundamental frequency (Hz)
A1 = 1   # Amplitude of fundamental
A2 = 0.5 # Amplitude of 2nd harmonic
A3 = 0.3 # Amplitude of 3rd harmonic

# Generate the signals
fundamental = A1 * np.sin(2 * np.pi * f1 * t)
second_harmonic = A2 * np.sin(2 * np.pi * 2 * f1 * t)
third_harmonic = A3 * np.sin(2 * np.pi * 3 * f1 * t)
combined_signal = fundamental + second_harmonic + third_harmonic

# Plotting
plt.figure(figsize=(12, 10))

# Fundamental frequency
plt.subplot(4, 1, 1)
plt.plot(t, fundamental, label="Fundamental (50 Hz)", color="blue")
plt.title("Fundamental Frequency and Harmonics")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Second harmonic
plt.subplot(4, 1, 2)
plt.plot(t, second_harmonic, label="Second Harmonic (100 Hz)", color="orange")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Third harmonic
plt.subplot(4, 1, 3)
plt.plot(t, third_harmonic, label="Third Harmonic (150 Hz)", color="green")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Combined signal
plt.subplot(4, 1, 4)
plt.plot(t, combined_signal, label="Combined Signal", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Adjust layout and show
plt.tight_layout()
plt.show()
