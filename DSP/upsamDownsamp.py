import numpy as np
import matplotlib.pyplot as plt

fs = 4000
t = np.arange(0, 1, 1/fs)
original_signal = np.sin(2 * np.pi * 30 * t)
downsampling_factor = 3

downsampled_signal = original_signal[::downsampling_factor]
downsampled_time = t[::downsampling_factor]
upsampling_factor = 3
upsampled_signal = np.repeat(downsampled_signal,upsampling_factor)[:len(original_signal)]
upsampled_time = np.linspace(0, 1, len(upsampled_signal))

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, original_signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(3, 1, 2)
plt.stem(downsampled_time, downsampled_signal, basefmt='k', linefmt='r-',markerfmt='ro')
plt.title('Downsampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(3, 1, 3)
plt.plot(upsampled_time, upsampled_signal, 'g-')
plt.title('Upsampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()