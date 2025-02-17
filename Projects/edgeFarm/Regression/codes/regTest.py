import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\dataset\data4.csv")  # Replace with your actual dataset path


# Select features and target
X = data[['Temperature', 'Humidity']]  # Features (current day's temp and humidity)
y = data[['nextemperature', 'nexthumidity']]  # Target (next day's temp and humidity)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model for both temperature and humidity prediction
model_temp = LinearRegression()
model_temp.fit(X_train[['Temperature', 'Humidity']], y_train['nextemperature'])

model_humid = LinearRegression()
model_humid.fit(X_train[['Temperature', 'Humidity']], y_train['nexthumidity'])

# Save trained models
joblib.dump(model_temp, r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\Regression\models\temperature_model.pkl')
joblib.dump(model_humid, r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\Regression\models\humidity_model.pkl')

# Extract model coefficients and intercepts
coeff_temp = model_temp.coef_  # Coefficients for temperature prediction
intercept_temp = model_temp.intercept_  # Intercept for temperature prediction

coeff_humid = model_humid.coef_  # Coefficients for humidity prediction
intercept_humid = model_humid.intercept_  # Intercept for humidity prediction

print("Temperature model coefficients:", coeff_temp)
print("Temperature model intercept:", intercept_temp)

print("Humidity model coefficients:", coeff_humid)
print("Humidity model intercept:", intercept_humid)

# Save coefficients and intercepts to a text file
with open(r'C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\edgeFarm\Regression\models\model_coefficients.txt', 'w') as file:
    file.write("Temperature model coefficients: {}\n".format(coeff_temp))
    file.write("Temperature model intercept: {}\n".format(intercept_temp))
    file.write("Humidity model coefficients: {}\n".format(coeff_humid))
    file.write("Humidity model intercept: {}\n".format(intercept_humid))
