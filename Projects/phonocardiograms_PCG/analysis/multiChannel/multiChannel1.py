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

# Create a multi-channel signal (necessary for ICA)
def create_multichannel(signal):
    acknowledge("Creating pseudo-multichannel signal for ICA")
    delayed_signal = np.roll(signal, shift=100)  # Create a delayed version
    mixed_sources = np.vstack((signal, delayed_signal)).T  # Stack as 2D array
    return mixed_sources

# Perform FastICA for Blind Source Separation
def apply_fastica(mixed_signal, components=2):
    acknowledge("Performing FastICA for Blind Source Separation")

    # Ensure FastICA does not exceed available components
    ica = FastICA(n_components=components, random_state=42)
    
    separated_sources = ica.fit_transform(mixed_signal)  # Apply ICA

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
    plt.plot(time[:len(fetal)], fetal, label="Extracted Fetal Heart Sound", color='red')
    plt.title("Extracted Fetal Heart Sound (FastICA)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time[:len(maternal)], maternal, label="Extracted Maternal Heart Sound", color='green')
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

    # Create a multi-channel input for ICA
    mixed_signal = create_multichannel(filtered_signal)

    # Apply FastICA for Blind Source Separation
    fetal_signal, maternal_signal = apply_fastica(mixed_signal)

    # Plot the signals
    plot_signals(original_signal, filtered_signal, fetal_signal, maternal_signal)

    # Save the extracted fetal heart sound
    save_audio(fetal_signal, "extracted_fetal_sound.wav")

# Run the process
audio_path = r"C:\Users\kezin\Downloads\dataset\maternal\m23.wav"  # Replace with actual file path
process_fetal_heart_sound(audio_path)
