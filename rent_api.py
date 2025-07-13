from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load training data to get feature columns
df = pd.read_csv('bhopal_rent_data.csv')
model = joblib.load('rent_prediction_model.pkl')
scaler = joblib.load('rent_scaler.pkl')

# Normalize column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Normalize values
df['location'] = df['location'].str.strip().str.lower()
df['furnishing'] = df['furnishing'].str.strip().str.lower()
df['property_type'] = df['property_type'].str.strip().str.lower()

# Apply one-hot encoding to match training format
df = pd.get_dummies(df, columns=["location", "furnishing", "property_type"])
feature_columns = df.drop('rent', axis=1).columns.tolist()

# Define input schema
class RentFeatures(BaseModel):
    bhk: int
    bathrooms: int
    furnishing: str
    property_type: str
    location: str

@app.post("/predict/rent")
def predict_rent(features: RentFeatures):
    # Clean and format input
    furnishing = features.furnishing.strip().lower()
    property_type = features.property_type.strip().lower()
    location = features.location.strip().lower()

    # Initialize input with all zeros
    input_data = {col: 0 for col in feature_columns}
    input_data["bhk"] = features.bhk
    input_data["bathrooms"] = features.bathrooms

    warnings = []

    # One-hot encode values
    f_col = f"furnishing_{furnishing}"
    p_col = f"property_type_{property_type}"
    l_col = f"location_{location}"

    if f_col in input_data:
        input_data[f_col] = 1
    else:
        warnings.append(f"Unknown furnishing: {furnishing}")
    if p_col in input_data:
        input_data[p_col] = 1
    else:
        warnings.append(f"Unknown property type: {property_type}")
    if l_col in input_data:
        input_data[l_col] = 1
    else:
        warnings.append(f"Unknown location: {location}")

    # Predict
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    return {
        "predicted_rent": round(float(prediction[0]), 2),
    }
