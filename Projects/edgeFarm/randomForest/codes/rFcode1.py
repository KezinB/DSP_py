import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\dataset\data2.csv")

# Select features (Ignore Date column)
X = data[['Soil Moisture', 'Temperature', 'Humidity']]
y = data['Watering Decision']  # 0 = Do Not Water, 1 = Water

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy * 100:.2f}%')

# Save trained model
joblib.dump(model, r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\randomForest\models\rf_model.pkl")

# Extract tree rules for C++ implementation
feature_importances = model.feature_importances_
print("Feature Importances:", feature_importances)

# Extract a single decision tree from the Random Forest
tree_rules = export_text(model.estimators_[0], feature_names=['Soil Moisture', 'Temperature', 'Humidity'])
print(tree_rules)