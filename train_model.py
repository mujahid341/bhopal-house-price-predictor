import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# load dataset
df = pd.read_csv('bhopal_house_price.csv')

# clean the dataset
df = df.dropna() # remove rows with missing values
df = df.drop_duplicates() # remove duplicate rows

# convert location(categorical data) to numerical values
# this is done using one-hot encoding
df = pd.get_dummies(df, columns=['location'])

# separate features and label(target variable)
x = df.drop('price', axis=1)  # features
y = df['price'] # label

# now scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, shuffle=True)

#create a linear regression model
model = LinearRegression()      

# train the model
model.fit(x_train, y_train)

# make predictions on the test set
y_pred = model.predict(x_test)  

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# joblib.dump(model, 'house_price_model.pkl')  # save the model
# joblib.dump(scaler, 'scaler.pkl')  # save the scaler

# # train the model using Random Forest Regressor
# # create a random forest regressor model
# random_forest_model = RandomForestRegressor(n_estimators=200, random_state=42)  

# #train the model
# random_forest_model.fit(x_train, y_train)   

# # make predictions on the test set
# y_pred_rf = random_forest_model.predict(x_test)     
# # evaluate the model
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf) 
# print(f'Random Forest Mean Squared Error: {mse_rf}')
# print(f'Random Forest R-squared: {r2_rf}')
