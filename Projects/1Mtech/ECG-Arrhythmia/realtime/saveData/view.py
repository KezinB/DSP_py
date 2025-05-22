import serial
ser = serial.Serial('COM6', 9600)  # Replace 'COM3' with your port
with open('data.txt', 'w') as f:
    while True:
        line = ser.readline().decode().strip()
        f.write(line + '\n')
        print(line)