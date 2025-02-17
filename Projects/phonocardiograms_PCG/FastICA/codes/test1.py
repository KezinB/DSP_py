import numpy as np
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA
import librosa

# Load two mixed audio signals
rate1, mixed1 = wav.read(r"C:\Users\kezin\Downloads\dataset\fetal\f112.wav")
rate2, mixed2 = wav.read(r"C:\Users\kezin\Downloads\dataset\maternal\m112.wav")

# Ensure same sampling rate and stack signals
assert rate1 == rate2  
# # mixed_signals = np.column_stack((mixed1, mixed2))
# max_length = max(len(mixed1), len(mixed2))
# mixed1 = np.pad(mixed1, (0, max_length - len(mixed1)), mode='constant')
# mixed2 = np.pad(mixed2, (0, max_length - len(mixed2)), mode='constant')
# mixed_signals = np.column_stack((mixed1, mixed2))

mixed1_resampled = librosa.resample(mixed1.astype(float), orig_sr=rate1, target_sr=rate2)
mixed2_resampled = librosa.resample(mixed2.astype(float), orig_sr=rate2, target_sr=rate1)

min_length = min(len(mixed1_resampled), len(mixed2_resampled))
mixed_signals = np.column_stack((mixed1_resampled[:min_length], mixed2_resampled[:min_length]))


# Apply FastICA
ica = FastICA(n_components=2)
separated_signals = ica.fit_transform(mixed_signals)  # This extracts the independent components

# Save separated audio files
wav.write(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\FastICA\Audio\separated_audio_f1.wav", rate1, separated_signals[:, 0].astype(np.int16))
wav.write(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\FastICA\Audio\separated_audio_m2.wav", rate2, separated_signals[:, 1].astype(np.int16))
