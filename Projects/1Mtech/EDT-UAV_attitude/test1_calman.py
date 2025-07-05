import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
import serial
from collections import deque

# Serial setup (adjust port)
ser = serial.Serial('COM8', 115200, timeout=1)

# Kalman Filter Initialization
dt = 0.01  # 100Hz sampling
Q = np.eye(4) * 0.01   # Process noise
R_base = np.eye(2) * 0.1    # Measurement noise (base)
P = np.eye(4)          # Covariance matrix
x = np.zeros((4, 1))   # State [roll, pitch, roll_bias, pitch_bias]

# State transition matrix
F = np.array([[1, 0, -dt, 0],
              [0, 1, 0, -dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Measurement matrix
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Buffers for plotting
buf_len = 1000
time_buf = deque(maxlen=buf_len)
roll_buf = deque(maxlen=buf_len)
pitch_buf = deque(maxlen=buf_len)
raw_roll_buf = deque(maxlen=buf_len)
raw_pitch_buf = deque(maxlen=buf_len)

def parse_serial_line(line):
    """Parse and validate a serial line into 6 floats."""
    parts = line.strip().split(',')
    if len(parts) != 6:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None

def adaptive_kf(accel, gyro):
    global x, P, Q
    # Adaptive noise tuning (simplified)
    acc_magnitude = np.linalg.norm(accel)
    if abs(acc_magnitude - 9.8) > 2.0:  # High acceleration
        R = R_base * 5.0  # Trust accelerometer less
    else:
        R = R_base.copy()  # Default

    # PREDICT STEP
    u = gyro[:2]  # Gyro x,y for roll/pitch
    x_pred = F @ x + np.array([[u[0]*dt], [u[1]*dt], [0], [0]])
    P_pred = F @ P @ F.T + Q

    # UPDATE STEP
    Z = np.array([[np.arctan2(accel[1], accel[2])],  # Roll from accel
                  [np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))]])  # Pitch

    y = Z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(4) - K @ H) @ P_pred

    x[:] = x_new
    P[:] = P_new

    return x[0,0], x[1,0]  # Return roll, pitch

# Real-time plotting setup
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
line_kf_roll, = ax1.plot([], [], 'b', label='KF Roll')
line_raw_roll, = ax1.plot([], [], 'r--', label='Raw Roll')
ax1.set_ylabel('Roll (deg)')
ax1.legend()
line_kf_pitch, = ax2.plot([], [], 'g', label='KF Pitch')
line_raw_pitch, = ax2.plot([], [], 'm--', label='Raw Pitch')
ax2.set_ylabel('Pitch (deg)')
ax2.set_xlabel('Time (s)')
ax2.legend()
plt.tight_layout()

t = 0

try:
    while True:
        try:
            line = ser.readline().decode(errors='ignore')
        except serial.SerialException:
            print("Serial error. Exiting.")
            break

        values = parse_serial_line(line)
        if values is None:
            continue

        accel = np.array(values[:3])
        gyro = np.array(values[3:])

        # Get raw attitude from accelerometer
        raw_roll = np.arctan2(accel[1], accel[2])
        raw_pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))

        # Kalman filter update
        roll, pitch = adaptive_kf(accel, gyro)

        # Update buffers
        t += dt
        time_buf.append(t)
        roll_buf.append(np.degrees(roll))
        pitch_buf.append(np.degrees(pitch))
        raw_roll_buf.append(np.degrees(raw_roll))
        raw_pitch_buf.append(np.degrees(raw_pitch))

        # Print angles
        print(f"Raw Roll: {np.degrees(raw_roll):.2f}°, Raw Pitch: {np.degrees(raw_pitch):.2f}° | "
              f"KF Roll: {np.degrees(roll):.2f}°, KF Pitch: {np.degrees(pitch):.2f}°")

        # Update plots efficiently
        line_kf_roll.set_data(time_buf, roll_buf)
        line_raw_roll.set_data(time_buf, raw_roll_buf)
        ax1.relim()
        ax1.autoscale_view()

        line_kf_pitch.set_data(time_buf, pitch_buf)
        line_raw_pitch.set_data(time_buf, raw_pitch_buf)
        ax2.relim()
        ax2.autoscale_view()

        plt.pause(0.001)

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    ser.close()
    plt.ioff()
    plt.show()