import os
import numpy as np
import librosa
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Paths to directories
fetal_folder = r"C:\Users\kezin\Downloads\dataset\fetal"
noise_folder = r"C:\Users\kezin\Downloads\dataset\maternal"
model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\final\accu\SVM\svm_model.pkl"

# Acknowledgment messages
def acknowledge(message):
    print(f"=== {message} ===")

# Feature extraction from audio
def extract_features(audio_path, sr=16000, n_mfcc=5):  # Reduced MFCCs to 5
    try:
        signal, _ = librosa.load(audio_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)  # Extract MFCCs
        features = np.mean(mfccs, axis=1)  # Mean of MFCCs
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Phase 1: Collecting data
acknowledge("Starting data collection")

X = []  # Features
y = []  # Labels

# Load fetal sound files
for file in os.listdir(fetal_folder):
    file_path = os.path.join(fetal_folder, file)
    features = extract_features(file_path)
    if features is not None:
        X.append(features)
        y.append(1)  # Label: 1 for fetal sounds

# Load noise sound files
for file in os.listdir(noise_folder):
    file_path = os.path.join(noise_folder, file)
    features = extract_features(file_path)
    if features is not None:
        X.append(features)
        y.append(0)  # Label: 0 for noise

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

acknowledge("Data collection and splitting completed")

# Phase 2: Training the SVM model
acknowledge("Starting model training")

scaler = StandardScaler()  # Standardize features
X_train_scaled = scaler.fit_transform(X_train)

# Use a simpler model (linear kernel instead of RBF)
svm_model = SVC(kernel='linear', probability=True, random_state=42)  # Linear kernel
svm_model.fit(X_train_scaled, y_train)

acknowledge("Model training completed")

# Save the model
with open(model_save_path, 'wb') as model_file:
    pickle.dump({'model': svm_model, 'scaler': scaler}, model_file)

acknowledge(f"Model saved at {model_save_path}")

# Phase 3: Testing the model with noisy data
acknowledge("Starting model testing with noisy data")

# Add Gaussian noise to the test set
noise_level = 2  # Increased noise level
X_test_noisy = X_test + noise_level * np.random.normal(0, 1, X_test.shape)

# Standardize the noisy test set
X_test_noisy_scaled = scaler.transform(X_test_noisy)

# Evaluate the model on the noisy test set
y_pred_noisy = svm_model.predict(X_test_noisy_scaled)
accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
print(f"Accuracy on noisy test set: {accuracy_noisy:.2f}")

# Generate classification report
report = classification_report(y_test, y_pred_noisy, target_names=["Noise", "Fetal Sound"])
print("Classification Report:")
print(report)