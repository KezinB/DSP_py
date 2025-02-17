import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from PIL import Image
import os
from datetime import datetime

# Step 1: Data Preparation
# Load the MNIST dataset from keras
print("Loading the MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the data
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

print("Data loaded and normalized successfully.")

# Step 2: Model Implementation
# Initialize the KNN model
print("Initializing the KNN model...")
knn = KNeighborsClassifier(n_neighbors=3)

print("Training the KNN model...")
# Train the KNN model
knn.fit(X_train, y_train)

# Save the trained model with the current date and time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
joblib_file = f"C:\\Users\\kezin\\OneDrive\\Documents\\Codes\\python\\DSP\\lab\\models\\knn_model_{current_time}.pkl"
joblib.dump(knn, joblib_file)
print(f"Model saved to {joblib_file}")

# Load the saved model
knn = joblib.load(joblib_file)
print("Model loaded successfully.")

# Step 3: Model Evaluation
# Test the KNN model on the MNIST test dataset
print("Testing the KNN model...")
y_pred = knn.predict(X_test)

# Evaluate the accuracy
print("Evaluating the accuracy...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 4: Visualization
# Plot a few examples of the test images with their predicted labels
def plot_examples(images, labels, predictions, num_examples=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {labels[i]}\nPred: {predictions[i]}")
        plt.axis('off')
    plt.show()

plot_examples(X_test[:5], y_test[:5], y_pred[:5])

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
plt.show()

# Step 5: Test with Your Own Images and Save Preprocessed Images
def preprocess_and_save_image(image_path, save_dir):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img).astype('float32') / 255.0  # Normalize pixel values
    img = img.reshape(1, -1)  # Flatten the image
    
    # Save preprocessed image
    preprocessed_image_path = os.path.join(save_dir, os.path.basename(image_path))
    Image.fromarray((img.reshape(28, 28) * 255).astype('uint8')).save(preprocessed_image_path)
    
    return img

# Directory containing test images
test_images_dir = r"C:\Users\kezin\OneDrive\Documents\Codes\python\DSP\lab\testImages"
# Directory to save preprocessed images
preprocessed_images_dir = r"C:\Users\kezin\OneDrive\Documents\Codes\python\DSP\lab\preprocessedImages"

# Create directory if not exists
if not os.path.exists(preprocessed_images_dir):
    os.makedirs(preprocessed_images_dir)

# Iterate through each image in the directory and save preprocessed images
for filename in os.listdir(test_images_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(test_images_dir, filename)
        own_image = preprocess_and_save_image(image_path, preprocessed_images_dir)
        
        # Make prediction
        own_image_pred = knn.predict(own_image)
        print(f"Image: {filename}, Predicted label: {own_image_pred[0]}")
