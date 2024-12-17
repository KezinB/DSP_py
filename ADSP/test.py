import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 4  # Down-sampling factor
N = 1024  # Number of points for FFT
fs = 2 * np.pi  # Original sampling frequency (normalized to 2π)

# Generate a signal with multiple frequencies
frequencies = [0.5 * np.pi, 1.5 * np.pi, 2.5 * np.pi,3 * np.pi,3.5 * np.pi]  # Frequencies in the signal
amplitudes = [1, 0.7, 0.5]  # Corresponding amplitudes
#amplitudes = [0.7] 
n = np.arange(0, N)
signal = sum(a * np.sin(f * n) for a, f in zip(amplitudes, frequencies))

# FFT of the original signal
spectrum = np.abs(np.fft.fft(signal, N))
freqs = np.fft.fftfreq(N, d=1/fs)  # Frequency range

# Extend frequency axis for visualization (-6π to 6π)
freq_range = np.linspace(-6 * np.pi, 6 * np.pi, 12 * N)  # Extended frequency axis
spectrum_extended = np.tile(spectrum, 12)  # Repeat spectrum for extended range

# Down-sampling
downsampled_signal = signal[::M]
spectrum_downsampled = np.abs(np.fft.fft(downsampled_signal, N // M))
freqs_downsampled = np.fft.fftfreq(N // M, d=1/(fs/M))

# Visualization
plt.figure(figsize=(12, 8))

# Original signal spectrum
plt.subplot(2, 1, 1)
plt.plot(freq_range, spectrum_extended, label="Original Spectrum", color="blue")
plt.title("Frequency Spectrum: Original Signal")
plt.xlabel("Frequency (radians/sample)")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()

# Downsampled signal spectrum
plt.subplot(2, 1, 2)
plt.stem(freqs_downsampled, spectrum_downsampled, basefmt=" ", linefmt="red", markerfmt="ro", label="Downsampled Spectrum")
plt.title(f"Frequency Spectrum After {M}-Fold Decimation")
plt.xlabel("Frequency (radians/sample)")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
