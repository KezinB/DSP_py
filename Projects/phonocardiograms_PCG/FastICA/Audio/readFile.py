import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Load the separated audio files
rate1, audio1 = wav.read(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\FastICA\Audio\separated_audio_m2.wav")
rate2, audio2 = wav.read(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\FastICA\Audio\separated_audio_f1.wav")
rate3, audio3 = wav.read(r"C:\Users\kezin\Downloads\dataset\maternal\m112.wav")
rate4, audio4 = wav.read(r"C:\Users\kezin\Downloads\dataset\fetal\f112.wav")
# Create time axis
time1 = np.linspace(0, len(audio1) / rate1, num=len(audio1))
time2 = np.linspace(0, len(audio2) / rate2, num=len(audio2))
time3 = np.linspace(0, len(audio3) / rate3, num=len(audio3))
time4 = np.linspace(0, len(audio4) / rate4, num=len(audio4))

# Plot waveforms
plt.figure(figsize=(12, 5))

plt.subplot(4, 1, 2)
plt.plot(time1, audio1, color='b')
plt.title("Separated Audio 1")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 4)
plt.plot(time2, audio2, color='r')
plt.title("Separated Audio 2")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 1)
plt.plot(time3, audio3, color='b')
plt.title("Audio 1")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 3)
plt.plot(time4, audio4, color='r')
plt.title("Audio 2")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
