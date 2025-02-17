import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
TF_ENABLE_ONEDNN_OPTS=0
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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

# Step 3: Model Evaluation
# Test the KNN model
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
