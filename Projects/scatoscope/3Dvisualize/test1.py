import serial
import time
import math
import trimesh
import pythreejs as pjs
import numpy as np
from pythreejs import *
from IPython.display import display
from scipy.spatial.transform import Rotation as R

# Serial port settings
port = 'COM18'  # Change this to your Bluetooth COM port
baud_rate = 115200

# Open serial port with exception handling
try:
    ser = serial.Serial(port, baud_rate, timeout=1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

# Function to calculate orientation using accelerometer and gyroscope data
def calculate_orientation(ax, ay, az, gx, gy, gz, dt):
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))

    roll_angle = (0.98 * (roll + math.radians(gx * dt))) + (0.02 * roll)
    pitch_angle = (0.98 * (pitch + math.radians(gy * dt))) + (0.02 * pitch)

    return math.degrees(roll_angle), math.degrees(pitch_angle)  # Convert to degrees

# Function to create 3D scene from STL file
def create_scene(model_file):
    # Load the STL model
    mesh = trimesh.load_mesh(model_file)

    # STL files do not have faces, only triangular meshes
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32).flatten()
    print(f"Model vertices count: {len(mesh.vertices)}")
    print(f"Model faces count: {len(mesh.faces)}")

    # Convert to pythreejs geometry
    geometry = pjs.BufferGeometry(
        attributes={
            'position': pjs.BufferAttribute(vertices, normalized=False),
            'index': pjs.BufferAttribute(faces, normalized=False)
        }
    )

    material = pjs.MeshStandardMaterial(color='gray', metalness=0.5, roughness=0.5)
    model = pjs.Mesh(geometry=geometry, material=material)

    # Initialize the scene
    scene = Scene(children=[
        model,
        AmbientLight(color='#ffffff', intensity=0.75)
    ])

    # Adjust camera position
    camera = PerspectiveCamera(position=[10, 10, 10], up=[0, 0, 1], children=[
        DirectionalLight(color='#ffffff', position=[5, 5, 5], intensity=1.0)
    ])

    renderer = Renderer(camera=camera, scene=scene, controls=[OrbitControls(controlling=camera)], width=800, height=600)

    return scene, renderer, model

# Use your STL file path
model_file = r"C:\Users\kezin\OneDrive\Documents\business_ideas\EMPTSPACE\STL\9tailed_Lowpoly.stl"

# Load the STL model and create the scene
scene, renderer, model = create_scene(model_file)

# Display the renderer (only once)
display(renderer)

# Loop to read data from sensor and update model orientation
while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue  # Skip if empty data

        data = line.split(',')
        if len(data) != 6:
            print(f"Invalid data received: {line}")
            continue  # Ignore corrupted data

        # Ensure the model is in a visible range
        model.scale = [0.5, 0.5, 0.5]

        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = map(float, data)
        dt = 0.25  # Delay time in seconds (corresponding to Arduino delay)

        roll_angle, pitch_angle = calculate_orientation(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt)

        # Convert Euler angles to quaternion
        rotation_quaternion = R.from_euler('xyz', [roll_angle, pitch_angle, 0], degrees=True).as_quat()
        model.quaternion = tuple(rotation_quaternion)

        # Render the scene and update the model orientation
        renderer.render(scene, renderer.camera)

        time.sleep(dt)

    except (ValueError, serial.SerialException) as e:
        print(f"Error: {e}")
        continue  # Skip faulty readings

ser.close()

