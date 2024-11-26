import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample

# Function to extract colored noise based on alpha
def extract_colored_noise(input_noise, alpha, fs):
    N = len(input_noise)
    freqs = fftfreq(N, d=1/fs)
    spectrum = fft(input_noise)
    # Modify the spectrum for the desired power-law relationship
    modifier = np.abs(freqs)**(alpha / 2)
    modifier[0] = 0  # Avoid division by zero at DC
    modified_spectrum = spectrum * modifier
    # Transform back to time domain
    return np.real(ifft(modified_spectrum))

# Parameters
fs = 16000  # Sampling frequency for recording
duration = 10  # Duration of recording in seconds

# Record audio
print("Recording for 10 seconds...")
recorded_audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float64')
sd.wait()  # Wait until recording is finished
recorded_audio = recorded_audio.flatten()  # Convert to 1D array
print("Recording complete!")

# Resample if needed (optional)
# recorded_audio = resample(recorded_audio, len(recorded_audio))

# Save raw audio
folder_path = "C:\\Users\\HP\\OneDrive\\Documents\\Codes\\python\\ResearchRaw\\ColouredNoise\\recExtract"
now = datetime.now()
folder_name = now.strftime("%Y_%m_%d-%H_%M_%S_recExtractNoise")
full_folder_path = os.path.join(folder_path, folder_name)
os.makedirs(full_folder_path, exist_ok=True)

raw_audio_path = os.path.join(full_folder_path, "raw_audio.wav")
write(raw_audio_path, fs, recorded_audio)
print(f"Raw audio saved at: {raw_audio_path}")

# Time vector
N = len(recorded_audio)
time = np.linspace(0, N/fs, N)

# Define alpha values and labels
alphas = [-2, -1, 0, 1, 2]
labels = [
    "Violet Noise (α=-2)",
    "Blue Noise (α=-1)",
    "White Noise (α=0)",
    "Pink Noise (α=1)",
    "Brown Noise (α=2)"
]
colors = ["purple", "blue", "gray", "pink", "brown"]

# Extract colored noise components
extracted_noises = [extract_colored_noise(recorded_audio, alpha, fs) for alpha in alphas]

# Save extracted noises to CSV and audio files
for alpha, noise, label in zip(alphas, extracted_noises, labels):
    # Save CSV
    csv_filename = os.path.join(full_folder_path, f"noise_alpha_{alpha}.csv")
    df = pd.DataFrame({"Time (s)": time, "Amplitude": noise})
    df.to_csv(csv_filename, index=False)
    
    # Save the noise as audio
    noise_audio_path = os.path.join(full_folder_path, f"noise_alpha_{alpha}.wav")
    write(noise_audio_path, fs, noise.astype(np.float32))  # Save as .wav
    print(f"Saved {label} as {noise_audio_path}")

# Save the plot
plt.figure(figsize=(12, 15))

# Plot original audio
plt.subplot(len(alphas) + 1, 1, 1)
plt.plot(time, recorded_audio, color='black', alpha=0.7, label="Recorded Audio")
plt.title("Original Recorded Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Plot extracted components
for i, (noise, label, color) in enumerate(zip(extracted_noises, labels, colors), start=2):
    plt.subplot(len(alphas) + 1, 1, i)
    plt.plot(time, noise, color=color, alpha=0.7, label=label)
    plt.title(label)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

plt.tight_layout()
plot_filename = os.path.join(full_folder_path, "noise_analysis_plot.png")
plt.savefig(plot_filename)

# Show the plot
plt.show()

# Save the code itself to the same folder
script_file_path = os.path.join(full_folder_path, "combined_code.py")
with open(script_file_path, "w") as code_file:
    code_file.write(open(__file__).read())

print(f"Analysis completed. Outputs saved in {full_folder_path}")
