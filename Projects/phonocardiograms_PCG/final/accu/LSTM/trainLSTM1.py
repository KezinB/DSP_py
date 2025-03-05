import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Paths to the dataset
fetal_folder = r"C:\Users\kezin\Downloads\dataset\SVM\Aug_Fetal"
noise_folder = r"C:\Users\kezin\Downloads\dataset\SVM\Aug_Noise"
model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\final\accu\LSTM\lstm_model.h5"

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

# Save the model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Add noise to the test set
noise_level = 0.2  # Adjust noise level as needed
X_test_noisy = X_test + noise_level * np.random.normal(0, 1, X_test.shape)

# Evaluate the model on the noisy test set
y_pred = model.predict(X_test_noisy)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Accuracy on noisy test set: {accuracy:.2f}")

# Generate classification report
report = classification_report(y_test_classes, y_pred_classes, target_names=["Noise", "Fetal Sound"])
print("Classification Report:")
print(report)