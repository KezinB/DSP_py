import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\scatoscope\datas\logs\ScatoscopeData_2025-02-17_18-46-12.csv')

# Plotting the Accelerometer data
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.plot(df['Time'], df['Accel X'], label='Accel X')
plt.plot(df['Time'], df['Accel Y'], label='Accel Y')
plt.plot(df['Time'], df['Accel Z'], label='Accel Z')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.title('Accelerometer Data')

# Plotting the Gyroscope data
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.plot(df['Time'], df['Gyro X'], label='Gyro X')
plt.plot(df['Time'], df['Gyro Y'], label='Gyro Y')
plt.plot(df['Time'], df['Gyro Z'], label='Gyro Z')
plt.xlabel('Time')
plt.ylabel('Gyroscope')
plt.legend()
plt.title('Gyroscope Data')

plt.tight_layout()
plt.show()
