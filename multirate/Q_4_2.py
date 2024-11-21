from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

def qmf_filter_bank(audio):
    # Define QMF filter coefficients
    h0 = np.array([-0.05441584, 0.31287159, 0.67563074, 0.31287159, -0.05441584])
    h1 = np.array([0.0038097, -0.035147, 0.0584279, 0.5159484, 0.6998308, 
                   0.5159484, 0.0584279, -0.035147, 0.0038097])

    # Apply QMF filter bank
    v0 = np.convolve(audio, h0, mode='same')
    v1 = np.convolve(audio, h1, mode='same')
    v2 = np.convolve(audio, -h0, mode='same')  # High-pass filter
    v3 = np.convolve(audio, -h1, mode='same')  # High-pass filter

    return v0, v1, v2, v3

def plot_tones_and_channels(tones, channels, sample_rate):
    # Plot the tones
    plt.figure(figsize=(12, 6))
    for i, tone in enumerate(tones):
        plt.subplot(2, 2, i + 1)
        t = np.arange(len(tone)) / sample_rate
        plt.plot(t, tone)
        plt.title(f'Tone {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    # Plot the channels
    plt.figure(figsize=(12, 6))
    t_channels = np.arange(len(channels)) / sample_rate
    plt.plot(t_channels, channels)
    plt.title('Channels')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.show()

# Input file path
input_file_path = 'C:\\Users\\HP\\Downloads\\alert-102266.wav'

# Load the audio file
audio = AudioSegment.from_wav(input_file_path)
sample_rate = audio.frame_rate
channels = np.array(audio.get_array_of_samples()).astype(np.float64).flatten()

# Normalize the channels
channels /= np.max(np.abs(channels))

# Apply the QMF filter bank
tones = qmf_filter_bank(channels)

# Plot the tones and channels
plot_tones_and_channels(tones, channels, sample_rate)

# Export and play the original audio
print("Playing Original Audio...")
audio.export("original_audio.wav", format="wav")

# Export and play each extracted tone
for i, tone in enumerate(tones):
    tone_audio = AudioSegment(
        tone.astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # Set sample width to 2 bytes (16 bits)
        channels=1
    )
    tone_audio.export(f"tone_{i + 1}.wav", format="wav")
    print(f"Playing Extracted Tone {i + 1}...")
