import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import librosa

# Feature Extraction Function
def extract_features(signal, sr, frame_size=1024, hop_size=512):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, hop_length=hop_size)
    zcr = librosa.feature.zero_crossing_rate(y=signal, hop_length=hop_size)
    energy = np.sum(np.square(signal)) / len(signal)
    return np.concatenate([mfcc.mean(axis=1), [zcr.mean(), energy]])

# Load Dataset
def load_dataset(file_paths, labels):
    features, label_list = [], []
    for file_path, label in zip(file_paths, labels):
        signal, sr = librosa.load(file_path, sr=None)
        feature = extract_features(signal, sr)
        features.append(feature)
        label_list.append(label)
    return np.array(features), np.array(label_list)

if __name__ == "__main__":
    # Training Dataset (Replace with actual file paths)
    fetal_file_paths = [
        "path_to_fetal_audio1.wav",
        "path_to_fetal_audio2.wav",
        "path_to_fetal_audio3.wav"
    ]  # Add your fetal audio file paths
    maternal_file_paths = [
        "path_to_maternal_audio1.wav",
        "path_to_maternal_audio2.wav",
        "path_to_maternal_audio3.wav"
    ]  # Add your maternal audio file paths

    # Labels (1 = fetal heart sound, 0 = maternal heart sound or noise)
    fetal_labels = [1] * len(fetal_file_paths)
    maternal_labels = [0] * len(maternal_file_paths)

    # Combine fetal and maternal data
    file_paths = fetal_file_paths + maternal_file_paths
    labels = fetal_labels + maternal_labels

    # Load dataset
    features, labels = load_dataset(file_paths, labels)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train SVM Classifier
    svm = SVC(kernel="linear", C=1.0)
    svm.fit(X_train, y_train)

    # Validate the Model
    y_pred = svm.predict(X_test)
    print(f"Training Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save the Model
    dump(svm, "svm_model.joblib")
    print("SVM model saved to 'svm_model.joblib'")
