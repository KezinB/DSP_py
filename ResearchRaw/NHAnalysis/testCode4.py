import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.fftpack import fft, fftfreq
from scipy.optimize import minimize, newton

# Step 1: Generate a clean sine wave
fs = 500      # Sample rate
f = 5         # Frequency of the sine wave
x = np.arange(fs)
clean_signal = np.sin(2 * np.pi * f * x / fs)

# Step 2: Add noise to the sine wave
noise = np.random.normal(0, 0.5, clean_signal.shape)
noisy_signal = clean_signal + noise

# Step 3: Filter the noisy signal
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

cutoff = 7  # Desired cutoff frequency of the filter, Hz
filtered_signal = butter_lowpass_filter(noisy_signal, cutoff, fs, order=6)

# Step 4: Perform FFT
fft_clean = np.fft.fft(clean_signal)
fft_noisy = np.fft.fft(noisy_signal)
fft_filtered = np.fft.fft(filtered_signal)
frequencies = np.fft.fftfreq(len(clean_signal), 1/fs)

# Step 5: Steepest Descent Method (example for optimization)
def objective_function(params):
    # Simple objective function for illustration
    a, b = params
    return np.sum((a * noisy_signal + b - clean_signal)**2)

initial_guess = [1, 0]
result = minimize(objective_function, initial_guess, method='BFGS')
optimized_params = result.x

# Step 6: Newton's Method (example for finding roots)
def function_to_solve(x):
    return x**2 - 2  # Example function

root = newton(function_to_solve, 1.0)

# Step 7: Amplitude Convergence (simple check)
amplitude_convergence = np.max(np.abs(filtered_signal))

# Step 8: Sequential Reduction (simplify problem)
# Example: Sequentially reduce noise level
reduced_noise_signal = noisy_signal
for i in range(5):
    reduced_noise_signal = butter_lowpass_filter(reduced_noise_signal, cutoff, fs, order=6)

# Plot the time domain results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(x, clean_signal, label='Clean Sine Wave')
plt.plot(x, noisy_signal, label='Noisy Sine Wave', color='orange')
plt.plot(x, filtered_signal, label='Filtered Sine Wave', color='green')
plt.title('Time Domain Analysis')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# Plot the frequency domain results
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fft_clean), label='Clean Sine Wave FFT')
plt.plot(frequencies, np.abs(fft_noisy), label='Noisy Sine Wave FFT', color='orange')
plt.plot(frequencies, np.abs(fft_filtered), label='Filtered Sine Wave FFT', color='green')
plt.title('Frequency Domain Analysis')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
