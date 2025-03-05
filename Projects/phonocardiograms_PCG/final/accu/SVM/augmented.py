import os
import numpy as np
import librosa
import soundfile as sf  # Import soundfile for saving audio
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Paths to directories
fetal_folder = r"C:\Users\kezin\Downloads\dataset\fetal"
noise_folder = r"C:\Users\kezin\Downloads\dataset\maternal"
augmented_folder = r"C:\Users\kezin\Downloads\dataset\SVM\augmented_data"
model_save_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\final\accu\SVM\svm_model.pkl"

# Create the augmented data folder if it doesn't exist
if not os.path.exists(augmented_folder):
    os.makedirs(augmented_folder)

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

# Function to generate synthetic white noise
def generate_white_noise(length, amplitude=1.0):
    """
    Generate white noise.
    :param length: Length of the noise signal.
    :param amplitude: Amplitude of the noise (controls loudness).
    :return: White noise signal.
    """
    return amplitude * np.random.normal(0, 1, length)

# Function to augment data by adding noise
def augment_data(signal, noise_level=0.5):
    """
    Augment data by adding synthetic white noise.
    :param signal: Original signal.
    :param noise_level: Amplitude of the noise.
    :return: Augmented signal.
    """
    noise = generate_white_noise(len(signal), amplitude=noise_level)
    return signal + noise

# Phase 1: Collecting and augmenting data
acknowledge("Starting data collection and augmentation")

X = []  # Features
y = []  # Labels

# Load fetal sound files and augment
for file in os.listdir(fetal_folder):
    file_path = os.path.join(fetal_folder, file)
    signal, sr = librosa.load(file_path, sr=16000)
    features = extract_features(file_path)
    if features is not None:
        X.append(features)
        y.append(1)  # Label: 1 for fetal sounds
        # Augment data to create 10 samples per file
        for i in range(9):  # Generate 9 additional samples
            augmented_signal = augment_data(signal, noise_level=0.5)
            augmented_file_path = os.path.join(augmented_folder, f"aug_fetal_{file}_{i}.wav")
            sf.write(augmented_file_path, augmented_signal, sr)  # Use soundfile to save audio
            augmented_features = extract_features(augmented_file_path)
            if augmented_features is not None:
                X.append(augmented_features)
                y.append(1)

# Load noise sound files and augment
for file in os.listdir(noise_folder):
    file_path = os.path.join(noise_folder, file)
    signal, sr = librosa.load(file_path, sr=16000)
    features = extract_features(file_path)
    if features is not None:
        X.append(features)
        y.append(0)  # Label: 0 for noise
        # Augment data to create 10 samples per file
        for i in range(9):  # Generate 9 additional samples
            augmented_signal = augment_data(signal, noise_level=0.5)
            augmented_file_path = os.path.join(augmented_folder, f"aug_noise_{file}_{i}.wav")
            sf.write(augmented_file_path, augmented_signal, sr)  # Use soundfile to save audio
            augmented_features = extract_features(augmented_file_path)
            if augmented_features is not None:
                X.append(augmented_features)
                y.append(0)

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

acknowledge("Data collection, augmentation, and splitting completed")

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
noise_level = 0.5  # Increased noise level
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