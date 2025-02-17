import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import FastICA
import soundfile as sf

# Acknowledgment messages
def acknowledge(message):
    print(f"=== {message} ===")

# Load audio signal
def load_audio(file_path, sr=16000):
    acknowledge(f"Loading audio file: {file_path}")
    signal, _ = librosa.load(file_path, sr=sr)
    return signal

# Apply Savitzky-Golay filter for noise reduction
def apply_savgol_filter(signal, window_size=5, poly_order=3):
    acknowledge("Applying Savitzky-Golay filter for noise reduction")
    return savgol_filter(signal, window_length=window_size, polyorder=poly_order)

# Perform FastICA for Blind Source Separation
def apply_fastica(mixed_signal, components=2):
    acknowledge("Performing FastICA for Blind Source Separation")

    # Reshape signal for FastICA (ensure it's treated as a 2D array)
    mixed_signal = mixed_signal.reshape(-1, 1)  

    # Ensure FastICA does not exceed available components
    ica = FastICA(n_components=min(components, mixed_signal.shape[1]), random_state=42)
    
    separated_sources = ica.fit_transform(mixed_signal)  # Apply ICA

    # Handle case where FastICA extracts only one component
    if separated_sources.shape[1] < 2:
        acknowledge("Warning: Only one independent component found. Returning single component.")
        return separated_sources[:, 0], np.zeros_like(separated_sources[:, 0])  # Return zero for second source

    return separated_sources[:, 0], separated_sources[:, 1]  # Return separated components

# Plot the signals
def plot_signals(original, filtered, fetal, maternal, sr=16000):
    time = np.linspace(0, len(original) / sr, num=len(original))
    
    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.plot(time, original, label="Original Signal", color='black')
    plt.title("Original PCG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time, filtered, label="Filtered Signal (Savitzky-Golay)", color='blue')
    plt.title("Filtered Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time, fetal, label="Extracted Fetal Heart Sound", color='red')
    plt.title("Extracted Fetal Heart Sound (FastICA)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time, maternal, label="Extracted Maternal Heart Sound", color='green')
    plt.title("Extracted Maternal Heart Sound (FastICA)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Save extracted fetal heart sound
def save_audio(signal, file_name, sr=16000):
    acknowledge(f"Saving extracted fetal heart sound: {file_name}")
    sf.write(file_name, signal, sr)

# Main function
def process_fetal_heart_sound(audio_file):
    # Load mixed PCG signal
    original_signal = load_audio(audio_file)

    # Apply Savitzky-Golay filter
    filtered_signal = apply_savgol_filter(original_signal)

    # Apply FastICA for Blind Source Separation
    fetal_signal, maternal_signal = apply_fastica(filtered_signal)

    # Plot the signals
    plot_signals(original_signal, filtered_signal, fetal_signal, maternal_signal)

    # Save the extracted fetal heart sound
    save_audio(fetal_signal,"extracted_fetal_sound.wav")

# Run the process
audio_path = r"C:\Users\kezin\Downloads\dataset\maternal\m23.wav"  # Replace with actual file path
process_fetal_heart_sound(audio_path)
