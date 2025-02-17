import tensorflow as tf
import pandas as pd

# Load dataset
data = pd.read_csv(r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\dataset\data1.csv')
X = data[['soil_moisture', 'temp', 'humidity']]
y = data['watering_decision']

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50)

# ✅ Save the model in HDF5 (.h5) format
model.save(r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\models\model1.h5')

print("Model saved successfully as model1.h5")
