import streamlit as st
import pandas as pd
import joblib

# Load models and scalers
price_model = joblib.load('house_price_model.pkl')
price_scaler = joblib.load('scaler.pkl')

rent_model = joblib.load('rent_prediction_model.pkl')
rent_scaler = joblib.load('rent_scaler.pkl')

# Load and prepare training data to get features
price_df = pd.read_csv('bhopal_house_price.csv')
price_df = pd.get_dummies(price_df, columns=["location"])
price_features = price_df.drop('price', axis=1).columns.tolist()
price_locations = sorted([col.replace("location_", "") for col in price_features if col.startswith("location_")])

rent_df = pd.read_csv('bhopal_rent_data.csv')
rent_df.columns = rent_df.columns.str.strip().str.replace(" ", "_").str.lower()
rent_df['location'] = rent_df['location'].str.strip().str.lower()
rent_df['furnishing'] = rent_df['furnishing'].str.strip().str.lower()
rent_df['property_type'] = rent_df['property_type'].str.strip().str.lower()
rent_df = pd.get_dummies(rent_df, columns=["location", "furnishing", "property_type"])
rent_features = rent_df.drop('rent', axis=1).columns.tolist()
rent_locations = sorted([col.replace("location_", "") for col in rent_features if col.startswith("location_")])
furnishing_options = sorted([col.replace("furnishing_", "") for col in rent_features if col.startswith("furnishing_")])
property_types = sorted([col.replace("property_type_", "") for col in rent_features if col.startswith("property_type_")])

# Streamlit App UI
st.title("Bhopal Real Estate Prediction")

#select the mode what you want to predict
mode = st.selectbox("Select Prediction Type", ["House Price", "Rent"])

if mode == "House Price":
    st.subheader("House Price Prediction")
    with st.form(key='house_form'):
        bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
        sqft = st.number_input("Total Square Feet", min_value=100.0, max_value=10000.0, step=50.0)
        location = st.selectbox("Location", price_locations)
        submit = st.form_submit_button("Predict Price")

    if submit:
        input_dict = {
            "bhk": bhk,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "total_sqft": sqft,
            f"location_{location}": 1
        }

        model_input = {col: input_dict.get(col, 0) for col in price_features}
        input_df = pd.DataFrame([model_input])
        scaled_input = price_scaler.transform(input_df)
        prediction = price_model.predict(scaled_input)
        st.success(f"Estimated House Price: ₹ {round(prediction[0], 2):,}")

elif mode == "Rent":
    st.subheader("Rent Prediction")

    with st.form(key='rent_form'):
        bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1, key='rent_bhk')
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1, key='rent_bath')
        furnishing = st.selectbox("Furnishing", furnishing_options)
        property_type = st.selectbox("Property Type", property_types)
        location = st.selectbox("Location", rent_locations)
        submit = st.form_submit_button("Predict Rent")

    if submit:
        # Prepare input data
        input_dict = {
            "bhk": bhk,
            "bathrooms": bathrooms,
            f"furnishing_{furnishing}": 1,
            f"property_type_{property_type}": 1,
            f"location_{location}": 1
        }

        model_input = {col: input_dict.get(col, 0) for col in rent_features}
        input_df = pd.DataFrame([model_input])
        scaled_input = rent_scaler.transform(input_df)
        prediction = rent_model.predict(scaled_input)
        st.success(f"Estimated Monthly Rent: ₹ {round(prediction[0], 2):,}")
