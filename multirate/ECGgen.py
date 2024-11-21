import neurokit2 as nk
import matplotlib.pyplot as plt

# Generate a synthetic ECG signal
# Define parameters: duration (in seconds), sampling rate (Hz), and heart rate (beats per minute)
duration = 50  # 10 seconds
sampling_rate = 500  # 500 Hz sampling rate
heart_rate = 60  # 60 beats per minute

# Create the ECG signal using NeuroKit2
ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate)

# Plot the ECG signal
plt.figure(figsize=(10, 4))
plt.plot(ecg_signal)
plt.title("Synthetic ECG Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
