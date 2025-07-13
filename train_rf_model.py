import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('bhopal_rent_data.csv')

# Rename inconsistent column names for safety
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Columns should now be: 'location', 'bhk', 'bathrooms', 'furnishing', 'property_type', 'rent'

# Lowercase string values in categorical columns
df['location'] = df['location'].str.lower()
df['furnishing'] = df['furnishing'].str.lower()
df['property_type'] = df['property_type'].str.lower()

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['location', 'furnishing', 'property_type'])

# Clean the dataset
df = df.dropna()
df = df.drop_duplicates()

# Separate features and label
X = df.drop('rent', axis=1)
y = df['rent']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.4f}')

# Save model and scaler
joblib.dump(model, 'rent_prediction_model.pkl')
joblib.dump(scaler, 'rent_scaler.pkl')

# train using linear regression as well
# from sklearn.linear_model import LinearRegression   
# # Create a linear regression model
# linear_model = LinearRegression()
# # Train the linear regression model
# linear_model.fit(X_train, y_train)
# # Make predictions with linear regression
# y_pred_linear = linear_model.predict(X_test)
# # Evaluate linear regression model
# mse_linear = mean_squared_error(y_test, y_pred_linear)
# r2_linear = r2_score(y_test, y_pred_linear)
# print(f'Linear Regression Mean Squared Error: {mse_linear:.2f}')
# print(f'Linear Regression R-squared: {r2_linear:.4f}')