import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\dataset\data2.csv")  # Ensure CSV contains Soil Moisture, Temperature, Humidity, Watering Decision

# Select features (Ignore Date column)
X = data[['Soil Moisture', 'Temperature', 'Humidity']]
y = data['Watering Decision']  # Target variable (0 = Do Not Water, 1 = Water)

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier (Gaussian Naive Bayes for continuous data)
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Extract mean, variance, and class priors for the C++ implementation
means = model.theta_  # Means for each class
variances = model.var_  # Variances for each feature
priors = model.class_prior_  # Priors for each class (not log)

# Print out the values (this is for the C++ implementation)
print("Means:", means)
print("Variances:", variances)
print("Priors:", priors)

# Save trained model
joblib.dump(model, r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\naveBais\model\nb_model.pkl")

# Save the extracted data to a text file
with open(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\naveBais\coEffi\nb_model_data.txt", "w") as file:
    file.write("Naive Bayes Model Parameters:\n")
    file.write("\nMeans:\n")
    for i, class_means in enumerate(means):
        file.write(f"Class {i} means: {class_means}\n")
    
    file.write("\nVariances:\n")
    for i, class_variances in enumerate(variances):
        file.write(f"Class {i} variances: {class_variances}\n")
    
    file.write("\nPriors:\n")
    for i, class_prior in enumerate(priors):
        file.write(f"Class {i} prior: {class_prior}\n")

print("Naive Bayes model data saved to nb_model_data.txt")
