import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Paths to the dataset
fetal_folder = r"C:\Users\kezin\Downloads\dataset\fetal"
noise_folder = r"C:\Users\kezin\Downloads\dataset\maternal"
model_save_path = r"C:\Users\kezin\Downloads\dataset\model\lstm_model.h5"

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

# Collecting data
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

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape data for LSTM (Adding batch dimension)
X = X[..., np.newaxis]  # Shape: (num_samples, n_mels, fixed_length, 1)

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=2)

# Define LSTM model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(2, activation='softmax')  # Output layer with 2 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# Save the model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Test the model with a new signal
test_signal_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\paperBased\audio\recovered_pcg1.wav"
test_mel_spec = extract_mel_spectrogram(test_signal_path)

if test_mel_spec is not None:
    # Reshape the test data for LSTM (add batch dimension)
    test_mel_spec = test_mel_spec[np.newaxis, ..., np.newaxis]  # Shape: (1, n_mels, fixed_length, 1)

    # Load trained model
    model = tf.keras.models.load_model(model_save_path)

    # Make prediction
    probabilities = model.predict(test_mel_spec)[0]
    prediction = np.argmax(probabilities)

    # Output prediction result
    if prediction == 1:
        print("Prediction: The test signal is a **Fetal Heart Sound**")
    else:
        print("Prediction: The test signal is **Noise**")

    print(f"Prediction Probabilities - Fetal Sound: {probabilities[1]:.2f}, Noise: {probabilities[0]:.2f}")
else:
    print("Test signal could not be processed.")
