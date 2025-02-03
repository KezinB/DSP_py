import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from joblib import load
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Savitzky-Golay Filter for Smoothing
def apply_savitzky_golay(signal, window_size=101, polyorder=3):
    return savgol_filter(signal, window_length=window_size, polyorder=polyorder)

# Bandpass Filtering
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

# Feature Extraction
def extract_features(signal, sr, frame_size=1024, hop_size=512):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, hop_length=hop_size)
    zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=hop_size)
    energy = np.sum(np.square(signal)) / len(signal)
    return np.concatenate([mfcc.mean(axis=1), [zcr.mean(), energy]])

# Predict and Reconstruct Signal
def predict_signal(signal, sr, svm):
    frame_size = int(0.1 * sr)  # 100ms frames
    hop_size = int(0.05 * sr)  # 50% overlap
    predictions = []
    enhanced_signal = np.zeros_like(signal)

    for i in range(0, len(signal) - frame_size, hop_size):
        segment = signal[i : i + frame_size]
        smoothed_segment = apply_savitzky_golay(segment)
        filtered_segment = bandpass_filter(smoothed_segment, 100, 300, sr)
        features = extract_features(filtered_segment, sr)
        pred = svm.predict([features])
        predictions.append(pred)

        # If predicted as fetal heart sound, retain the segment
        if pred == 1:
            enhanced_signal[i : i + frame_size] = segment

    return enhanced_signal

if __name__ == "__main__":
    # Load the Trained SVM Model
    svm = load("svm_model.joblib")
    print("SVM model loaded from 'svm_model.joblib'")

    # Load Test Signal
    test_signal_path = "path_to_test_signal.wav"  # Replace with actual test signal
    test_signal, sr = librosa.load(test_signal_path, sr=None)

    # Predict and Enhance the Signal
    enhanced_signal = predict_signal(test_signal, sr, svm)

    # Save Enhanced Signal as Audio
    enhanced_signal_path = "enhanced_signal.wav"
    write(enhanced_signal_path, sr, (enhanced_signal * 32767).astype(np.int16))
    print(f"Enhanced signal saved as '{enhanced_signal_path}'")

    # Plot Original and Enhanced Signal
    time = np.linspace(0, len(test_signal) / sr, num=len(test_signal))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, test_signal, label="Original Signal")
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, enhanced_signal, label="Enhanced Signal", color="red")
    plt.title("Enhanced Fetal Heart Sound Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.tight_layout()
    plt.show()
