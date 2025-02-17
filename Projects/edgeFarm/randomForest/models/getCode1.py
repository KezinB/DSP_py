from sklearn.tree import export_text
import joblib

# Load trained model
model = joblib.load(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\randomForest\models\rf_model.pkl")

# Print rules for **all** trees
for i, tree in enumerate(model.estimators_):  # Loop through all trees
    print(f"\nTree {i+1} Rules:")
    print(export_text(tree, feature_names=['Soil Moisture', 'Temperature', 'Humidity']))
