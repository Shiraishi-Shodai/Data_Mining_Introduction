import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the data
file_path = 'exe21.csv'
data = pd.read_csv(file_path)

# Split the data into training and testing sets
train_data = data[data['x'] < 3.5]
test_data = data[data['x'] > 3.6]

# Prepare the training and testing datasets
x_train = train_data['x'].values.reshape(-1, 1)
y_train = train_data['y'].values.reshape(-1, 1)
x_test = test_data['x'].values.reshape(-1, 1)
y_test = test_data['y'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train)
y_train_scaled = scaler.fit_transform(y_train)
x_test_scaled = scaler.transform(x_test)
y_test_scaled = scaler.transform(y_test)

# Reshape the data to fit LSTM input
x_train_scaled = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
x_test_scaled = x_test_scaled.reshape((x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train_scaled, y_train_scaled, epochs=100, batch_size=1, verbose=2)

# Make predictions
y_pred_scaled = model.predict(x_test_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Plot the training data, test data, and predictions
plt.figure(figsize=(12, 6))
plt.plot(x_train, y_train, label='Training Data', color='blue')
plt.plot(x_test, y_test, label='Test Data', color='green')
plt.plot(x_test, y_pred, label='Predictions', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('LSTM Model Predictions')
plt.show()
