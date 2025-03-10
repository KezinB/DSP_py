import os
import numpy as np
import librosa
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Paths to directories
fetal_folder = r"C:\Users\kezin\Downloads\dataset\fetal"
noise_folder = r"C:\Users\kezin\Downloads\dataset\maternal"
model_save_path = r"C:\Users\kezin\Downloads\dataset\model\svm_model.pkl"

# Acknowledgment messages
def acknowledge(message):
    print(f"=== {message} ===")

# Feature extraction from audio
def extract_features(audio_path, sr=16000):
    try:
        signal, _ = librosa.load(audio_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Extract 13 MFCCs
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

acknowledge("Data collection completed")

# Phase 2: Training the SVM model
acknowledge("Starting model training")

scaler = StandardScaler()  # Standardize features
X_scaled = scaler.fit_transform(X)

svm_model = SVC(kernel='rbf', probability=True, random_state=42)  # SVM classifier
svm_model.fit(X_scaled, y)

acknowledge("Model training completed")

# Save the model
with open(model_save_path, 'wb') as model_file:
    pickle.dump({'model': svm_model, 'scaler': scaler}, model_file)

acknowledge(f"Model saved at {model_save_path}")

# Phase 3: Testing the model
acknowledge("Starting model testing")

# Load a noise signal for testing
test_signal_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\paperBased\audio\recovered_pcg1.wav"
test_features = extract_features(test_signal_path)

if test_features is not None:
    # Standardize the features
    with open(model_save_path, 'rb') as model_file:
        model_data = pickle.load(model_file)
        svm_model = model_data['model']
        scaler = model_data['scaler']

    test_features_scaled = scaler.transform([test_features])
    prediction = svm_model.predict(test_features_scaled)[0]
    probabilities = svm_model.predict_proba(test_features_scaled)[0]

    # Output prediction and probabilities
    if prediction == 1:
        acknowledge("The test signal is predicted as Fetal Heart Sound")
    else:
        acknowledge("The test signal is predicted as Noise")

    print(f"Prediction Probabilities: Fetal Sound: {probabilities[1]:.2f}, Noise: {probabilities[0]:.2f}")

    # Plot the log-frequency power spectrogram of the test signal
    test_signal, sr = librosa.load(test_signal_path, sr=16000)

    # Compute the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=test_signal, sr=sr, n_mels=128)
    
    # Convert the Mel spectrogram to decibels (log-frequency scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot the log-frequency power spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Frequency Power Spectrogram - SVM model')

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

else:
    acknowledge("Test signal could not be processed")
