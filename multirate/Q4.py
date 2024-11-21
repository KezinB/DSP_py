import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

# Define the lowpass filter coefficients
B1 = [-0.006443977, 0.02745539, -0.00758164, -0.0913825, 0.09808522, -0.4807962]
h0 = np.concatenate((B1, np.flip(B1)))

# Generate the highpass filter coefficients
h1 = [(-1) ** k * coef for k, coef in enumerate(h0)]

# Load the original audio signal (convert to .wav if necessary)
Fs, x = wavfile.read("C:\\Users\\HP\\Downloads\\alert-102266.wav")  # Ensure this is a .wav file

# Create a time vector for the original signal
t = np.linspace(0, len(x) / Fs, len(x))

# Plot the original audio signal
plt.figure(1)
plt.plot(t, x)
plt.title('Original Audio Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Define analysis filters for 4 channels
g0, g1, g2, g3 = h0, [-coef for coef in h1], h0, [-coef for coef in h1]

# Analysis filters and additional subbands
v0, v1, v2, v3 = [signal.lfilter(g, 1, x)[::4] for g in [g0, g1, g2, g3]]

# Plot the analysis results
plt.figure(2, figsize=(10, 8))
for i, vi in enumerate([v0, v1, v2, v3], start=1):
    plt.subplot(4, 1, i)
    ti = np.linspace(0, len(vi) / (Fs / 4), len(vi))  # Use Fs / 4 for downsampled signals
    plt.plot(ti, vi)
    plt.ylabel(f'v{i}[n]')
plt.suptitle('Figure 2: Signal Analysis')
plt.xlabel('Time')
plt.show()

# Synthesis bank (upsampling by inserting zeros)
wi = [np.zeros(len(vi) * 4) for vi in [v0, v1, v2, v3]]
for i, (vi, w) in enumerate(zip([v0, v1, v2, v3], wi)):
    w[::4] = vi  # Upsample by inserting zeros

# Synthesis filters and plot the results
yi = [signal.lfilter(g, 1, w) for g, w in zip([g0, g1, g2, g3], wi)]
plt.figure(3, figsize=(10, 8))
for i, yi in enumerate(yi, start=1):
    plt.subplot(4, 1, i)
    ti = np.linspace(0, len(yi) / Fs, len(yi))  # Use original Fs for synthesized signals
    plt.plot(ti, yi)
    plt.ylabel(f'y{i}[n]')
plt.suptitle('Figure 3: Signal Synthesis')
plt.xlabel('Time')
plt.show()
