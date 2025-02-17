import tensorflow as tf
import numpy as np

# Load Keras model
keras_model = tf.keras.models.load_model(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\models\model1.h5")

# Convert model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset():
    for i in range(100):
        yield [np.array([[0.5, 25, 60]], dtype=np.float32)]
converter.representative_dataset = representative_dataset

# Ensure full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save model
with open(r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\models\model1_quant.tflite', 'wb') as f:
    f.write(tflite_model)
