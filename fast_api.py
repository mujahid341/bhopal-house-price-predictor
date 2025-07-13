from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load training data to extract feature columns
df = pd.read_csv('bhopal_house_price.csv')
df = pd.get_dummies(df, columns=["location"])
feature_columns = df.drop('price', axis=1).columns.tolist()

# Define input format
class HouseFeatures(BaseModel):
    bhk: int
    bathrooms: int
    bedrooms: int
    total_sqft: float
    location: str

@app.post('/predict')
def predict_price(features: HouseFeatures):
    # Build input dictionary with default 0s
    input_data = {col: 0 for col in feature_columns}

    # Fill known values
    input_data['bhk'] = features.bhk
    input_data['bathrooms'] = features.bathrooms
    input_data['bedrooms'] = features.bedrooms
    input_data['total_sqft'] = features.total_sqft

    # One-hot encode the location
    location_column = f"location_{features.location}"
    if location_column in input_data:
        input_data[location_column] = 1
    else:
        print(f"⚠️ Location '{features.location}' not found in training data. Prediction may be less accurate.")

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    return {"predicted_price": round(float(prediction[0]), 2)}
