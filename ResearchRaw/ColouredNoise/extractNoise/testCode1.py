import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import pandas as pd
import shutil

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
fs = 1000  # Sampling frequency in Hz
N = 10000  # Number of samples
time = np.linspace(0, N/fs, N)

# Generate or input the noise signal (replace this with your actual signal)
np.random.seed(42)
given_noise = np.random.randn(N)  # Example white noise signal

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
extracted_noises = [extract_colored_noise(given_noise, alpha, fs) for alpha in alphas]

# Create output directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"./Noise_Analysis_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# Save extracted noises to CSV
for alpha, noise, label in zip(alphas, extracted_noises, labels):
    csv_filename = f"{output_folder}/noise_alpha_{alpha}.csv"
    df = pd.DataFrame({"Time (s)": time, "Amplitude": noise})
    df.to_csv(csv_filename, index=False)

# Save the plot
plt.figure(figsize=(12, 15))

# Plot original noise
plt.subplot(len(alphas) + 1, 1, 1)
plt.plot(time, given_noise, color='black', alpha=0.7, label="Original Noise")
plt.title("Original Noise Signal")
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
plot_filename = f"{output_folder}/noise_analysis_plot.png"
plt.savefig(plot_filename)
plt.close()

# Save a copy of the script
script_filename = f"{output_folder}/script_copy.py"
shutil.copy(__file__, script_filename)  # Assumes this script is saved as a file

print(f"Analysis completed. Outputs saved in {output_folder}")
