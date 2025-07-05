import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Set up the serial connection (change 'COM3' to your Arduino's port)
ser = serial.Serial('COM6', 9600, timeout=1)
plt.close('all')  # Close any existing matplotlib windows

# Create a fixed-length deque to store ECG values (adjust max_len for time window)
max_len = 20000  # Number of points to display
ecg_data = deque([0] * max_len, maxlen=max_len)

# Create figure and axis for plotting
fig, ax = plt.subplots()
line, = ax.plot(ecg_data)
ax.set_title('Real-time ECG Monitoring')
ax.set_xlabel('Time')
ax.set_ylabel('ECG Value')

# Set fixed X-axis limits based on buffer size
ax.set_xlim(0, max_len-1)

def update(frame):
    # Read data from serial
    while ser.in_waiting:
        try:
            raw_data = ser.readline().decode().strip()
            print(f"Raw data: {raw_data}")  # Debugging line to see raw data    
            # Handle lead-off detection
            if raw_data == '!':
                print("Lead-off detected! Check electrodes.")
                return line,
            
            # Convert to integer and add to data buffer
            value = int(raw_data)
            ecg_data.append(value)
            
        except (UnicodeDecodeError, ValueError):
            continue  # Skip invalid data
    
    # Update plot data
    line.set_ydata(ecg_data)
    line.set_xdata(range(len(ecg_data)))
    
    # Dynamic Y-axis scaling
    if ecg_data:  # Only update if we have data
        current_min = min(ecg_data)
        current_max = max(ecg_data)
        margin = 5  # Adjust this value to change padding around signal
        
        # Prevent zero-range y-axis
        if current_min == current_max:
            current_min -= margin
            current_max += margin
        
        ax.set_ylim(current_min - margin, current_max + margin)
    
    return line,

# Set up animation
ani = FuncAnimation(fig, update, interval=10, blit=True)

plt.show()