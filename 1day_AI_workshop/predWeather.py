import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r'C:\Users\kezin\OneDrive\Documents\Codes\python\1day_AI_workshop\weatherHistory.csv')

# Drop rows with missing values
df = df.dropna()

# Simple features (only numerical)
features = ['Apparent Temperature (C)', 'Wind Speed (km/h)',
            'Wind Bearing (degrees)', 'Visibility (km)',
            'Loud Cover', 'Pressure (millibars)']

X = df[features]
y_temp = df['Temperature (C)']
y_hum = df['Humidity']

# Split dataset
X_train, X_test, y_temp_train, y_temp_test, y_hum_train, y_hum_test = train_test_split(
    X, y_temp, y_hum, test_size=0.2, random_state=42
)

# Train models
model_temp = LinearRegression().fit(X_train, y_temp_train)
model_hum = LinearRegression().fit(X_train, y_hum_train)

# ---- Take user input ----
print("Enter the following details:")

input_data = []
for feature in features:
    val = float(input(f"{feature}: "))
    input_data.append(val)

# Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=features)

# Predict
pred_temp = model_temp.predict(input_df)[0]
pred_hum = model_hum.predict(input_df)[0]

print(f"\n📈 Predicted Temperature: {pred_temp:.2f} °C")
print(f"💧 Predicted Humidity: {pred_hum:.2f}")
