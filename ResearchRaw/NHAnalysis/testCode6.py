import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.optimize import minimize, newton

# Generate a clean sine wave
fs = 500  # Sample rate
f = 20  # Frequency of the sine wave
x = np.arange(fs)
clean_signal = np.sin(2 * np.pi * f * x / fs)

# Add noise to the sine wave
noise = np.random.normal(0, 0.5, clean_signal.shape)
noisy_signal = clean_signal + noise

# Frequency Analysis using FFT
fft_noisy = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(noisy_signal), 1/fs)
initial_freq_idx = np.argmax(np.abs(fft_noisy))
initial_freq = frequencies[initial_freq_idx]
initial_phase = np.angle(fft_noisy[initial_freq_idx])

# Cost Function for Steepest Descent
def cost_function(params):
    freq, phase = params
    sine_wave = np.sin(2 * np.pi * freq * x / fs + phase)
    return np.sum((sine_wave - noisy_signal)**2)

# Steepest Descent Method
initial_guess = [initial_freq, initial_phase]
result = minimize(cost_function, initial_guess, method='BFGS')
optimized_freq, optimized_phase = result.x

# Retardation Method (Weighting Coefficient)
weighting_coefficient = 0.9
optimized_cost_function = lambda p: weighting_coefficient * cost_function(p)
final_result = minimize(optimized_cost_function, [optimized_freq, optimized_phase], method='BFGS')
final_freq, final_phase = final_result.x

# Newton's Method for Amplitude Convergence (replacing with minimize)
def amplitude_function(a):
    sine_wave = a * np.sin(2 * np.pi * final_freq * x / fs + final_phase)
    return np.sum((sine_wave - noisy_signal)**2)

initial_amplitude_guess = 1
amplitude_result = minimize(lambda a: amplitude_function(a[0]), [initial_amplitude_guess], method='BFGS')
amplitude = amplitude_result.x[0]

# Final Convergence
def final_convergence_function(params):
    freq, phase, a = params
    sine_wave = a * np.sin(2 * np.pi * freq * x / fs + phase)
    return np.sum((sine_wave - noisy_signal)**2)

final_params = minimize(final_convergence_function, [final_freq, final_phase, amplitude], method='BFGS').x
final_freq, final_phase, final_amplitude = final_params

# Generate the converged sine wave
converged_signal = final_amplitude * np.sin(2 * np.pi * final_freq * x / fs + final_phase)

# Plot the results
plt.figure(figsize=(12, 12))

# Plot Clean and Noisy Sine Waves
plt.subplot(3, 1, 1)
plt.plot(x, clean_signal, label='Clean Sine Wave')
plt.plot(x, noisy_signal, label='Noisy Sine Wave', color='orange')
plt.title('Clean and Noisy Sine Waves')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# Plot Converged Sine Wave and Noise
plt.subplot(3, 1, 2)
plt.plot(x, converged_signal, label='Converged Sine Wave', color='green')
plt.plot(x, noise, label='Noise', color='red')
plt.title('Converged Sine Wave and Noise')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# Plot FFT Analysis
plt.subplot(3, 1, 3)
plt.plot(frequencies, np.abs(fft_noisy), label='Noisy Sine Wave FFT', color='orange')
plt.plot(frequencies, np.abs(np.fft.fft(converged_signal)), label='Converged Sine Wave FFT', color='green')
plt.title('FFT Analysis')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
