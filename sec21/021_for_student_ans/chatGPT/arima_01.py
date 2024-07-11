import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the data
file_path = 'exe21.csv'
data = pd.read_csv(file_path)

# Split the data into training and testing sets
train_data = data[data['x'] < 3.5]
test_data = data[data['x'] > 3.6]

# Prepare the training and testing datasets
x_train = train_data['x']
y_train = train_data['y']
x_test = test_data['x']
y_test = test_data['y']

# Fit the ARIMA model on the training data
model = ARIMA(y_train, order=(5, 1, 0))
model_fit = model.fit()

# Make predictions on the test data
y_pred = model_fit.forecast(steps=len(x_test))

# Plot the training data, test data, and predictions
plt.figure(figsize=(12, 6))
plt.plot(x_train, y_train, label='Training Data', color='blue')
plt.plot(x_test, y_test, label='Test Data', color='green')
plt.plot(x_test, y_pred, label='Predictions', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('ARIMA Model Predictions')
plt.show()
