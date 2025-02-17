import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

# Path to the saved model
model_save_path = r"C:\Users\kezin\Downloads\dataset\model\lstm_model.h5"

# Path to the test audio file
test_signal_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\paperBased\audio\recovered_pcg1.wav"

# Function to extract Mel spectrogram from audio
def extract_mel_spectrogram(audio_path, sr=16000, n_mels=128, fixed_length=100):
    try:
        signal, _ = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale

        # Pad or truncate to match the model's input shape
        if mel_spec_db.shape[1] < fixed_length:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, fixed_length - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :fixed_length]

        return mel_spec_db
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Compute Log Spectrum of the signal
def compute_log_spectrum(audio_path, sr=16000, n_fft=2048, hop_length=512):
    # Load the audio signal
    signal, _ = librosa.load(audio_path, sr=sr)

    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # Compute the magnitude of the STFT
    magnitude = np.abs(D)

    # Convert the magnitude to a log scale (logarithmic power spectrum)
    log_spectrum = librosa.amplitude_to_db(magnitude, ref=np.max)

    return log_spectrum

# Extract Mel spectrogram from the test signal
test_mel_spec = extract_mel_spectrogram(test_signal_path)

if test_mel_spec is not None:
    # Reshape for LSTM (Add batch & channel dimensions)
    test_mel_spec = test_mel_spec[np.newaxis, ..., np.newaxis]  # Shape: (1, n_mels, fixed_length, 1)

    # Load trained model
    model = tf.keras.models.load_model(model_save_path)

    # Make prediction
    probabilities = model.predict(test_mel_spec)[0]
    prediction = np.argmax(probabilities)  # 0: Noise, 1: Fetal Heart Sound

    # Output prediction result
    if prediction == 1:
        print("Prediction: The test signal is a **Fetal Heart Sound**")
    else:
        print("Prediction: The test signal is **Noise**")

    print(f"Prediction Probabilities - Fetal Sound: {probabilities[1]:.2f}, Noise: {probabilities[0]:.2f}")

    # Plot the test Mel Spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(test_mel_spec.squeeze(), sr=16000, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Test Signal Mel Spectrogram")

    # Annotate the prediction probabilities on the graph
    prediction_text = "Fetal Heart Sound" if prediction == 1 else "Noise"
    probabilities_text = f"Fetal Sound: {probabilities[1]:.2f}, Noise: {probabilities[0]:.2f}"
    plt.annotate(f"Prediction: {prediction_text}\n{probabilities_text}", 
                 xy=(0.5, 0.9), xycoords="axes fraction", 
                 ha="center", va="center", 
                 fontsize=12, color="white", weight="bold", 
                 bbox=dict(facecolor='black', alpha=0.7))

    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.show()

    # Compute and plot the log spectrum of the test signal
    log_spectrum = compute_log_spectrum(test_signal_path)

    # Plot the log-frequency spectrum
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrum, sr=16000, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log Frequency Power Spectrum - LSTM")

    # Annotate the prediction probabilities on the log-spectrum graph
    plt.annotate(f"Prediction: {prediction_text}\n{probabilities_text}", 
                 xy=(0.5, 0.9), xycoords="axes fraction", 
                 ha="center", va="center", 
                 fontsize=12, color="white", weight="bold", 
                 bbox=dict(facecolor='black', alpha=0.7))

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

else:
    print("Test signal could not be processed.")
