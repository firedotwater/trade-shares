import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Load historical data of Apple shares
data = pd.read_csv('AAPL.csv','r')

# Extract relevant features
features = data[['Open', 'Close', 'Volume']]

# Define the target variable
target = data['Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Define and fit the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the value of the shares
prediction = regressor.predict(X_test)

# Compare the predicted value to the actual value
error = prediction - y_test

# Check if the error is within a certain threshold
threshold = 0.05
if abs(error.mean()) < threshold:
    print("The model predicts that the shares will not change significantly.")
else:
    if error.mean() > 0:
        print("The model predicts that the shares will increase in value.")
    else:
        print("The model predicts that the shares will decrease in value.")
