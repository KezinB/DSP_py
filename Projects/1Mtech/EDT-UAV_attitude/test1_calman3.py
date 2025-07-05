import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
import serial
from collections import deque
import time

# Serial setup (adjust port)
ser = serial.Serial('COM8', 115200, timeout=1)

# Kalman Filter Initialization
dt = 0.01  # 100Hz sampling
# Q = np.eye(4) * 0.01   # Process noise
# R_base = np.eye(2) * 0.1    # Measurement noise (base)
# P = np.eye(4)          # Covariance matrix
# Kalman Filter Initialization (updated values)
Q = np.diag([0.01, 0.01, 0.001, 0.001])   # Reduced bias process noise
R_base = np.eye(2) * 0.3                   # Increased base measurement noise
P = np.diag([0.1, 0.1, 0.01, 0.01])        # Tuned initial covariance
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
roll_plot_buf = deque(maxlen=buf_len)
pitch_plot_buf = deque(maxlen=buf_len)
raw_roll_buf = deque(maxlen=buf_len)
raw_pitch_buf = deque(maxlen=buf_len)
roll_error_buf = deque(maxlen=buf_len)   # Add this
pitch_error_buf = deque(maxlen=buf_len)  # Add this

# Moving average for final output
roll_buf = deque(maxlen=5)
pitch_buf = deque(maxlen=5)

def parse_serial_line(line):
    """Parse and validate a serial line into 6 floats."""
    parts = line.strip().split(',')
    if len(parts) != 6:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None

# def adaptive_kf(accel, gyro):
#     global x, P, Q
#     # Adaptive noise tuning (simplified)
#     acc_magnitude = np.linalg.norm(accel)
#     if abs(acc_magnitude - 9.8) > 2.0:  # High acceleration
#         R = R_base * 5.0  # Trust accelerometer less
#     else:
#         R = R_base.copy()  # Default
def adaptive_kf(accel, gyro):
    global x, P, Q
    # Improved adaptive noise tuning
    acc_magnitude = np.linalg.norm(accel)
    error = abs(acc_magnitude - 9.8)
    
    # Smooth scaling factor (1-5 range)
    scale_factor = 1 + min(4, max(0, error - 1.0)) 
    R = R_base * scale_factor

    # PREDICT STEP
    # Inside adaptive_kf()
    u = gyro[:2] - x[2:, 0]  # Subtract bias from gyro measurements
    x_pred = F @ x + np.array([[u[0]*dt], [u[1]*dt], [0], [0]])
    # u = gyro[:2]  # Gyro x,y for roll/pitch
    # x_pred = F @ x + np.array([[u[0]*dt], [u[1]*dt], [0], [0]])
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
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9))  # Add ax3
line_kf_roll, = ax1.plot([], [], 'b', label='KF Roll')
line_raw_roll, = ax1.plot([], [], 'r--', label='Raw Roll')
ax1.set_ylabel('Roll (deg)')
ax1.legend()
line_kf_pitch, = ax2.plot([], [], 'g', label='KF Pitch')
line_raw_pitch, = ax2.plot([], [], 'm--', label='Raw Pitch')
ax2.set_ylabel('Pitch (deg)')
ax2.set_xlabel('Time (s)')
ax2.legend()
# Error plot lines
line_roll_err, = ax3.plot([], [], 'c', label='Roll Error')
line_pitch_err, = ax3.plot([], [], 'y', label='Pitch Error')
ax3.set_ylabel('Error (deg)')
ax3.set_xlabel('Time (s)')
ax3.legend()
plt.tight_layout()

t = 0
last_time = time.time()
accel_prev = np.zeros(3)  # Add before main loop for low-pass filter

# --- Calibration Routine ---
print("Calibrating... keep IMU stationary")
accel_samples = []
gyro_samples_deg = []  # Store gyro in degrees

for _ in range(500):  # 5 seconds at 100Hz
    values = parse_serial_line(ser.readline().decode())
    if values:
        accel_samples.append(values[:3])
        gyro_samples_deg.append(values[3:])

# Accelerometer calibration: preserve gravity in z-axis
mean_accel = np.mean(accel_samples, axis=0)
accel_bias = mean_accel - [0, 0, 9.8]  # Keep 9.8 in z-axis

# Gyro calibration: convert to rad/s after bias removal
gyro_bias_deg = np.mean(gyro_samples_deg, axis=0)
gyro_bias = gyro_bias_deg * np.pi / 180.0  # Convert bias to rad/s

print(f"Calibration complete.")
print(f"Accel bias: {accel_bias}")
print(f"Gyro bias (deg/s): {gyro_bias_deg}")

# --- End Calibration Routine ---

# Set initial orientation from gravity vector
initial_roll = np.arctan2(0, 9.8)  # Should be 0 on flat surface
initial_pitch = np.arctan2(-0, 9.8)  # Should be 0
x[0,0] = initial_roll
x[1,0] = initial_pitch

try:
    while True:

        values = parse_serial_line(ser.readline().decode())
        if values is None:
            continue

        accel = np.array(values[:3])
        gyro_deg = np.array(values[3:])
        
        # Apply calibration
        accel_cal = accel - accel_bias
        gyro_rad = gyro_deg * np.pi / 180.0  # Convert to rad/s
        gyro_cal = gyro_rad - gyro_bias

        # Add after serial reading
        accel_lp = 0.7 * accel_cal + 0.3 * accel_prev  # Tune 0.7/0.3 ratio
        accel_prev = accel_cal.copy()

        # Get raw attitude from accelerometer
        raw_roll = np.arctan2(accel_lp[1], accel_lp[2])
        raw_pitch = np.arctan2(-accel_lp[0], np.sqrt(accel_lp[1]**2 + accel_lp[2]**2))

        # Kalman filter update
        roll, pitch = adaptive_kf(accel_lp, gyro_cal)

        # Update buffers for plotting
        t += dt
        time_buf.append(t)
        roll_plot_buf.append(np.degrees(roll))
        pitch_plot_buf.append(np.degrees(pitch))
        raw_roll_buf.append(np.degrees(raw_roll))
        raw_pitch_buf.append(np.degrees(raw_pitch))

        # Moving average for final output
        roll_buf.append(np.degrees(roll))
        pitch_buf.append(np.degrees(pitch))
        smoothed_roll = np.mean(roll_buf)
        smoothed_pitch = np.mean(pitch_buf)

        # Print angles (smoothed KF output)
        # Calculate instantaneous errors
        roll_error = np.degrees(roll) - np.degrees(raw_roll)
        pitch_error = np.degrees(pitch) - np.degrees(raw_pitch)

        print(f"Raw Roll: {np.degrees(raw_roll):.2f}°, Raw Pitch: {np.degrees(raw_pitch):.2f}° | "
              f"KF Roll: {smoothed_roll:.2f}°, KF Pitch: {smoothed_pitch:.2f}° | "
              f"Err Roll: {roll_error:.2f}°, Err Pitch: {pitch_error:.2f}°")

        # Update plots efficiently
        line_kf_roll.set_data(time_buf, roll_plot_buf)
        line_raw_roll.set_data(time_buf, raw_roll_buf)
        ax1.relim()
        ax1.autoscale_view()

        line_kf_pitch.set_data(time_buf, pitch_plot_buf)
        line_raw_pitch.set_data(time_buf, raw_pitch_buf)
        ax2.relim()
        ax2.autoscale_view()

        # Update error plot
        roll_error_buf.append(roll_error)
        pitch_error_buf.append(pitch_error)
        line_roll_err.set_data(time_buf, roll_error_buf)
        line_pitch_err.set_data(time_buf, pitch_error_buf)
        ax3.relim()
        ax3.autoscale_view()

        plt.pause(0.1)

        if len(roll_plot_buf) > 0 and len(raw_roll_buf) > 0:
            roll_kf_arr = np.array(roll_plot_buf)
            roll_raw_arr = np.array(raw_roll_buf)
            pitch_kf_arr = np.array(pitch_plot_buf)
            pitch_raw_arr = np.array(raw_pitch_buf)

            roll_rmse = np.sqrt(np.mean((roll_kf_arr - roll_raw_arr) ** 2))
            pitch_rmse = np.sqrt(np.mean((pitch_kf_arr - pitch_raw_arr) ** 2))

            print(f"Live RMSE (Roll): {roll_rmse:.3f} deg | Live RMSE (Pitch): {pitch_rmse:.3f} deg")

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    ser.close()
    plt.ioff()
    plt.show()

    # --- RMSE Calculation ---
    if len(roll_plot_buf) > 0 and len(raw_roll_buf) > 0:
        roll_kf_arr = np.array(roll_plot_buf)
        roll_raw_arr = np.array(raw_roll_buf)
        pitch_kf_arr = np.array(pitch_plot_buf)
        pitch_raw_arr = np.array(raw_pitch_buf)

        roll_rmse = np.sqrt(np.mean((roll_kf_arr - roll_raw_arr) ** 2))
        pitch_rmse = np.sqrt(np.mean((pitch_kf_arr - pitch_raw_arr) ** 2))

        print(f"\nRMSE (Roll): {roll_rmse:.3f} deg")
        print(f"RMSE (Pitch): {pitch_rmse:.3f} deg")