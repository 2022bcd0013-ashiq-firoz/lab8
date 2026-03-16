from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Paths to artifacts
model_path = "output/model-housing.pkl"
encoder_path = "output/encoder-housing.pkl"

# Global variables for model and encoder
model = None
encoder = None

def load_artifacts():
    global model, encoder
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
    else:
        print(f"Warning: Model or encoder not found. Paths: {model_path}, {encoder_path}")

# Initial load
load_artifacts()

app = FastAPI(
    title="California Housing Price Prediction API",
    description="FastAPI-based inference service for predicting housing prices in California",
    version="1.0.0"
)

class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

@app.post("/predict")
def predict_house_price(features: HousingFeatures):
    global model, encoder
    # Reload if model wasn't available at startup (optional but helpful)
    if model is None or encoder is None:
        load_artifacts()
        if model is None or encoder is None:
            return {"error": "Model or encoder not loaded. Please run training first."}
    
    # Prepare input data
    # Encode ocean_proximity string to numeric
    try:
        # The LabelEncoder expects a list/array
        ocean_proximity_encoded = encoder.transform([features.ocean_proximity])[0]
    except ValueError:
        return {"error": f"Invalid ocean_proximity value. Expected one of: {list(encoder.classes_)}"}

    input_data = np.array([[
        features.longitude,
        features.latitude,
        features.housing_median_age,
        features.total_rooms,
        features.total_bedrooms,
        features.population,
        features.households,
        features.median_income,
        ocean_proximity_encoded
    ]])

    prediction = model.predict(input_data)[0]

    return {
        "name": "Ashiq Firoz",
        "roll_no": "2022BCD0013",
        "predicted_median_house_value": float(prediction)
    }

@app.get("/")
def root():
    return {"message": "Housing Price Prediction API is running"}
