import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from scipy.signal import butter, lfilter  # <-- Add this

# ==== Configuration ====
# PORT = 'COM8'        # Change to your ESP32 serial port - usb
PORT = 'COM11' 
BAUD_RATE = 115200
BUFFER_SIZE = 500     # Number of points to show in real-time
# MAX_Y = 5000          # Adjust based on your EEG range
# MAX_Y = 15523 
# MAX_Y = 25000 
MAX_Y = 40000 
FS = 250              # <-- Set your sampling rate (Hz), adjust as needed

# ==== Bandpass Filter Setup ====
# def butter_bandpass(lowcut, highcut, fs, order=4):
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=1, highcut=50, fs=FS, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# ==== Setup ====
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
data_buffer = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)

# ==== Plot Setup ====
fig, ax = plt.subplots()
line, = ax.plot(range(BUFFER_SIZE), data_buffer)
ax.set_ylim(-MAX_Y, MAX_Y)
ax.set_title("Real-time EEG Visualization")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# ==== Animation Function ====
def update(frame):
    try:
        line_data = ser.readline().decode().strip()
        if line_data:
            value = int(line_data)
            data_buffer.append(value)
            line.set_ydata(data_buffer)
    except:
        pass
    return line,

ani = animation.FuncAnimation(fig, update, interval=1, blit=True)
plt.show()
