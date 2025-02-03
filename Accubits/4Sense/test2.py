import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize serial connection
ser = serial.Serial('COM9', baudrate=115200, timeout=10)  # Replace 'COM9' with your actual COM port

# Lists to store the data for each sensor
data_A0 = []
data_A1 = []
data_A2 = []
data_A3 = []

# Create a function to update the plot
def update(frame):
    line = ser.readline().decode("utf-8").strip()
    print(line)
    if line.startswith("A0 and A1:") or line.startswith("A2 and A3:"):
        try:
            if line.startswith("A0 and A1:"):
                # Extract and parse A0 and A1 data
                data_values = line.split(":")[1].strip().split("|")
                if len(data_values) == 2:
                    data_A0.append(float(data_values[0]))
                    data_A1.append(float(data_values[1]))
            elif line.startswith("A2 and A3:"):
                # Extract and parse A2 and A3 data
                data_values = line.split(":")[1].strip().split("|")
                if len(data_values) == 2:
                    data_A2.append(float(data_values[0]))
                    data_A3.append(float(data_values[1]))

            # Update the plot
            ax.cla()
            ax.plot(data_A0, label='Sensor A0', color='blue')
            ax.plot(data_A1, label='Sensor A1', color='red')
            ax.plot(data_A2, label='Sensor A2', color='green')
            ax.plot(data_A3, label='Sensor A3', color='orange')
            ax.set_title('Live Voltage Plot for Sensors')
            ax.set_xlabel('Time')
            ax.set_ylabel('Voltage (V)')
            ax.relim()
            ax.autoscale_view()
            ax.legend(loc='upper right')

        except ValueError:
            # Handle invalid float conversion
            pass

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Set titles and labels for axes
ax.set_title('Live Voltage Plot for Sensors')
ax.set_xlabel('Time')
ax.set_ylabel('Voltage (V)')
ax.legend()

# Create an animation to update the plot
ani = animation.FuncAnimation(fig, update, interval=30)  # Adjust interval based on data rate (in milliseconds)

# Show the plot
plt.tight_layout()
plt.show()
