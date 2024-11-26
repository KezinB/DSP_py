import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

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

cutoff = 7  # desired cutoff frequency of the filter, Hz
filtered_signal = butter_lowpass_filter(noisy_signal, cutoff, fs, order=6)

# Step 4: Plot the results
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(x, clean_signal, label='Clean Sine Wave')
plt.title('Clean Sine Wave')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(x, noisy_signal, label='Noisy Sine Wave', color='orange')
plt.title('Noisy Sine Wave')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(x, filtered_signal, label='Filtered Sine Wave', color='green')
plt.title('Filtered Sine Wave')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(x, clean_signal, label='Clean Sine Wave')
plt.plot(x, noisy_signal, label='Noisy Sine Wave', color='orange')
plt.plot(x, filtered_signal, label='Filtered Sine Wave', color='green')
plt.title('Combined Plot')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
