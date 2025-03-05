import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# Paths to the dataset
fetal_folder = r"C:\Users\kezin\Downloads\dataset\SVM\Aug_Fetal"
noise_folder = r"C:\Users\kezin\Downloads\dataset\SVM\Aug_Noise"
model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\final\accu\randForest\rf_model.pkl"  # Path to save the model

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
        X.append(mel_spec.flatten())  # Flatten the 2D Mel spectrogram into 1D
        y.append(1)  # Label 1 for fetal sounds

# Process noise files
for file in os.listdir(noise_folder):
    file_path = os.path.join(noise_folder, file)
    mel_spec = extract_mel_spectrogram(file_path)
    if mel_spec is not None:
        X.append(mel_spec.flatten())  # Flatten the 2D Mel spectrogram into 1D
        y.append(0)  # Label 0 for noise

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees

# Train the model
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, model_save_path)
print(f"Model saved at {model_save_path}")

# Add noise to the test set
noise_level = 0.5  # Adjust noise level as needed
X_test_noisy = X_test + noise_level * np.random.normal(0, 1, X_test.shape)

# Evaluate the model on the noisy test set
y_pred = rf_model.predict(X_test_noisy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on noisy test set: {accuracy:.2f}")

# Generate classification report
report = classification_report(y_test, y_pred, target_names=["Noise", "Fetal Sound"])
print("Classification Report:")
print(report)