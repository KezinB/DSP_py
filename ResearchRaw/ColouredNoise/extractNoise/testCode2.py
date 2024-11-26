import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import pandas as pd

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
fs = 5000  # Sampling frequency in Hz
N = 10000  # Number of samples
time = np.linspace(0, N/fs, N)

# np.random.normal(0, 0.5, N)

# Generate or input the noise signal (replace this with your actual signal)
np.random.seed(42)
# given_noise = np.random.randn(N)  # Example white noise signal
given_noise = np.random.normal(0, 0.5, N) 

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


folder_path = "C:\\Users\\HP\\OneDrive\\Documents\\Codes\\python\\ResearchRaw\\ColouredNoise\\extractNoise"
now = datetime.now()
folder_name = now.strftime("%Y_%m_%d-%H_%M_%S_extractNoise")
full_folder_path = os.path.join(folder_path, folder_name)
os.makedirs(full_folder_path, exist_ok=True)

# Save extracted noises to CSV
for alpha, noise, label in zip(alphas, extracted_noises, labels):
    csv_filename = os.path.join(full_folder_path, f"noise_alpha_{alpha}.csv")
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
plot_filename = os.path.join(full_folder_path, "noise_analysis_plot.png")
plt.savefig(plot_filename)

# Show the plot
plt.show()

# Save the code itself to the same folder
with open(os.path.join(full_folder_path, "combined_code.py"), "w") as code_file:
    code_file.write(open(__file__).read())

print(f"Analysis completed. Outputs saved in {full_folder_path}")
