import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import joblib

# Load dataset (Ensure CSV contains: Soil Moisture, Temperature, Humidity, Watering Decision)
data = pd.read_csv(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\dataset\data2.csv")

# Select features (Ignore Date column)
X = data[['Soil Moisture', 'Temperature', 'Humidity']]
y = data['Watering Decision']  # Target variable (0 = Do Not Water, 1 = Water)

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=3)  # Limit tree depth for efficiency
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\decisionTree\models\decision_tree_model.pkl")

# Export decision tree rules for C++
tree_rules = export_text(model, feature_names=['Soil Moisture', 'Temperature', 'Humidity'])
print(tree_rules)
