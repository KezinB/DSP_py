import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import wavio

# Set parameters
sample_rate = 44100
channels = 1
duration = 9
filename = 'recording.wav'

# Record audio
print('Recording...')
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float64')
sd.wait()
print('Recording finished.')

# Save recorded audio
wavio.write(filename, audio_data, sample_rate, sampwidth=3)

# Load audio data from the file
wav_obj = wavio.read(filename)
audio_data = wav_obj.data
sample_rate = wav_obj.rate

# Apply interpolation and decimation
interpolated_2x = np.interp(np.arange(0, len(audio_data), 0.5), np.arange(0, len(audio_data)), audio_data[:, 0])
interpolated_8x = np.interp(np.arange(0, len(audio_data), 1/8), np.arange(0, len(audio_data)), audio_data[:, 0])
decimated_2x = audio_data[::2, 0]
decimated_8x = audio_data[::8, 0]

# Plotting functions
def plot_audio(title_str, signal, fs, subplot_index, total_subplots):
    plt.subplot(total_subplots, 2, subplot_index)
    plt.plot(np.arange(len(signal)) / fs, signal)
    plt.title(f'{title_str} - Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(total_subplots, 2, subplot_index + 1)
    plt.specgram(signal, NFFT=1024, Fs=fs, noverlap=512)
    plt.title(f'{title_str} - Frequency Domain')

# Plot time-domain and frequency-domain for each case
plt.figure(figsize=(12, 18))
plot_audio('Original Audio', audio_data[:, 0], sample_rate, 1, 5)
plot_audio('2x Interpolated Audio', interpolated_2x, sample_rate * 2, 3, 5)
plot_audio('8x Interpolated Audio', interpolated_8x, sample_rate * 8, 5, 5)
plot_audio('2x Decimated Audio', decimated_2x, sample_rate // 2, 7, 5)
plot_audio('8x Decimated Audio', decimated_8x, sample_rate // 8, 9, 5)
plt.tight_layout()
plt.show()

# Play the original and processed audio
def play_and_wait(audio, fs, title_str, duration):
    print(f'Playing {title_str}')
    sd.play(audio, fs)
    sd.wait()

play_and_wait(audio_data[:, 0], sample_rate, 'Original Audio', duration)
play_and_wait(interpolated_2x, sample_rate * 2, '2x Interpolated Audio', duration * 2)
play_and_wait(interpolated_8x, sample_rate * 8, '8x Interpolated Audio', duration * 8)
play_and_wait(decimated_2x, sample_rate // 2, '2x Decimated Audio', duration // 2)
play_and_wait(decimated_8x, sample_rate // 8, '8x Decimated Audio', duration // 8)
