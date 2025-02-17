import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

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
knn = KNeighborsClassifier(n_neighbors=4)

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
    indices = np.random.choice(len(images), num_examples, replace=False)  # Randomly select indices
    for i, idx in enumerate(indices):
        plt.subplot(1, num_examples, i+1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {labels[idx]}\nPred: {predictions[idx]}")
        plt.axis('off')
    plt.show()

plot_examples(X_test, y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
plt.show()
