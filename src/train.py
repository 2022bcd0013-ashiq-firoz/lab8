import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os
import joblib

# Load dataset
print("Loading dataset")
data_path = 'data/housing.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

print("Preprocessing Dataset")
# Handle missing values in total_bedrooms
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# Encode categorical variable 'ocean_proximity'
le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# Split into X and y
# Features: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

print("Splitting train test data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor")
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=0)
model.fit(X_train, y_train)

# Save model and label encoder
model_filename = 'output/model-housing.pkl'
encoder_filename = 'output/encoder-housing.pkl'
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
joblib.dump(model, model_filename)
joblib.dump(le, encoder_filename)
print(f"Model saved to {model_filename}")
print(f"Encoder saved to {encoder_filename}")

r2_score_value = model.score(X_test, y_test)
print(f"R^2 Score: {r2_score_value:.2f}")

y_pred = model.predict(X_test)
mse_value = mean_squared_error(y_test, y_pred)
rmse_value = np.sqrt(mse_value)
print(f"Mean Squared Error (MSE): {mse_value:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_value:.2f}")

print("Saving metrics as a JSON output")
data = {
    "Experiment ID": "Exp-Housing-01",
    "Model Type": "Random Forest Regressor",
    "Train/Test-Split": "80-20",
    "MSE": float(mse_value),
    "RMSE": float(rmse_value),
    "accuracy": float(r2_score_value)
}

filename = 'output/metrics.json'

if os.path.exists(filename):
    with open(filename, 'r') as json_file:
        existing_data = json.load(json_file)
    
    if isinstance(existing_data, list):
        existing_data.append(data)
    else:
        existing_data = [existing_data, data]
    
    with open(filename, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)
else:
    with open(filename, 'w') as json_file:
        json.dump([data], json_file, indent=4)

print(f"Data successfully saved to {filename}")
