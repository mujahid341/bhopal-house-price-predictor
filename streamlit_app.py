import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load training data and prepare feature columns
df = pd.read_csv('bhopal_house_price.csv')
df = pd.get_dummies(df, columns=["location"])
feature_columns = df.drop('price', axis=1).columns.tolist()

# Extract location names from one-hot encoded columns
locations = sorted([col.replace("location_", "") for col in feature_columns if col.startswith("location_")])

# Streamlit UI
st.title("Bhopal House Price Prediction App")
st.write("Enter the house details to predict the estimated price.")

# form for user input
with st.form(key='house_features_form'):
    bhk = st.number_input("No. of BHK", min_value=1, max_value=10, step=1)
    bedrooms = st.number_input("No. of Bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("No. of Bathrooms", min_value=1, max_value=10, step=1)
    total_sqft = st.number_input("Total Square Feet", min_value=100.0, max_value=10000.0, step=50.0)
    location = st.selectbox("Select Location", options=locations)

    submit_button = st.form_submit_button(label='Predict Price')

if submit_button:
    # Prepare input dictionary
    input_data = {
        "bhk": bhk,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "total_sqft": total_sqft,
        f"location_{location}": 1
    }

    # Fill missing columns with 0
    model_input = {col: input_data.get(col, 0) for col in feature_columns}
    input_df = pd.DataFrame([model_input])

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict price
    prediction = model.predict(scaled_input)
    st.success(f"Estimated House Price: ₹ {round(prediction[0], 2):,}")
