import os
import numpy as np
import librosa
import librosa.display
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Paths
fetal_folder = r"C:\Users\kezin\Downloads\dataset\fetal"
noise_folder = r"C:\Users\kezin\Downloads\dataset\maternal"
model_save_path = r"C:\Users\kezin\Downloads\dataset\model\cnn_model.h5"

# Acknowledgment function
def acknowledge(message):
    print(f"=== {message} ===")

# Convert audio to Mel Spectrogram
def extract_mel_spectrogram(audio_path, sr=16000, n_mels=128):
    try:
        signal, _ = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Collecting data
acknowledge("Starting data collection")

X = []
y = []

# Process fetal heart sound files
for file in os.listdir(fetal_folder):
    file_path = os.path.join(fetal_folder, file)
    mel_spec = extract_mel_spectrogram(file_path)
    if mel_spec is not None:
        X.append(mel_spec)
        y.append(1)  # Label 1 for fetal sounds

# Process noise files
for file in os.listdir(noise_folder):
    file_path = os.path.join(noise_folder, file)
    mel_spec = extract_mel_spectrogram(file_path)
    if mel_spec is not None:
        X.append(mel_spec)
        y.append(0)  # Label 0 for noise

fixed_length = 500  # Adjust as needed

# Resize spectrograms
X = np.array([np.pad(mel, ((0, 0), (0, max(0, fixed_length - mel.shape[1]))), mode='constant')[:,:fixed_length] for mel in X])
y = np.array(y)

# Ensure all spectrograms have the same shape (padding if necessary)
max_len = max([mel.shape[1] for mel in X])
X = np.array([np.pad(mel, ((0, 0), (0, max_len - mel.shape[1])), mode='constant') for mel in X])

# Reshape for CNN (adding channel dimension)
X = X[..., np.newaxis]  # Shape: (num_samples, height, width, channels)

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=2)

acknowledge("Data collection completed")

# Define CNN model
acknowledge("Starting model training")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # Output layer with 2 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# Save the model
model.save(model_save_path)
acknowledge(f"Model saved at {model_save_path}")

# Load test signal
acknowledge("Starting model testing")

test_signal_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\paperBased\audio\recovered_pcg1.wav"
test_mel_spec = extract_mel_spectrogram(test_signal_path)

if test_mel_spec is not None:
    # Ensure test spectrogram matches training dimensions
    if test_mel_spec.shape[1] < max_len:
        test_mel_spec = np.pad(test_mel_spec, ((0, 0), (0, max_len - test_mel_spec.shape[1])), mode='constant')
    else:
        test_mel_spec = test_mel_spec[:, :max_len]  # Truncate if it's longer

    # Reshape for CNN (add batch and channel dimensions)
    test_mel_spec = test_mel_spec[np.newaxis, ..., np.newaxis]  # Shape: (1, 128, 100, 1)

    # Load trained model
    model = tf.keras.models.load_model(model_save_path)

    # Predict class
    probabilities = model.predict(test_mel_spec)[0]  # Now has the correct shape

    prediction = np.argmax(probabilities)

    # Output prediction
    if prediction == 1:
        acknowledge("The test signal is predicted as Fetal Heart Sound")
    else:
        acknowledge("The test signal is predicted as Noise")

    print(f"Prediction Probabilities: Fetal Sound: {probabilities[1]:.2f}, Noise: {probabilities[0]:.2f}")

    # Plot the test spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(test_mel_spec.squeeze(), sr=16000, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Test Signal Mel Spectrogram")
    plt.show()
else:
    acknowledge("Test signal could not be processed")
