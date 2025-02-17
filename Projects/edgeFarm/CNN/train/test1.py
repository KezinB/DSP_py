import tensorflow as tf
import pandas as pd

# Load dataset
data = pd.read_csv(r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\dataset\data1.csv')
X = data[['soil_moisture', 'temp', 'humidity']]
y = data['watering_decision']

# Simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\models\model1.tflite', 'wb') as f:
    f.write(tflite_model)