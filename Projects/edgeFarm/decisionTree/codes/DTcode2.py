import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\dataset\data2.csv")  # Ensure CSV contains Soil Moisture, Temperature, Humidity, Watering Decision

# Select features (Ignore Date column)
X = data[['Soil Moisture', 'Temperature', 'Humidity']]
y = data['Watering Decision']  # Target variable (0 = Do Not Water, 1 = Water)

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(
    max_depth=None,  # Allow the tree to grow without depth limit
    min_samples_split=2,  # Split as soon as there are 2 samples (this ensures deep trees)
    min_samples_leaf=1  # Allow nodes with 1 sample
)

model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\decisionTree\models\decision_tree_model.pkl")

# Export decision tree rules
tree_rules = export_text(model, feature_names=['Soil Moisture', 'Temperature', 'Humidity'])
print(tree_rules)
