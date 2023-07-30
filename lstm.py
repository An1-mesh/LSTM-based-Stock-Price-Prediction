import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Define the ticker symbol
ticker = 'BRK-A'

# Download the data
data = yf.download(ticker, start='2022-03-29', end='2023-03-29')

# Prepare the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Define the time steps and split the data into training and testing datasets
time_steps = 30
X_train, y_train = [], []
X_test, y_test = [], []

for i in range(time_steps, len(data_scaled)):
    if i < len(data_scaled) - time_steps:
        X_train.append(data_scaled[i - time_steps:i, 0])
        y_train.append(data_scaled[i, 0])
    else:
        X_test.append(data_scaled[i - time_steps:i, 0])
        y_test.append(data_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile and fit the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Inverse transform the scaled predictions and actual values
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the root mean squared error
rmse = np.sqrt(np.mean(((y_pred - y_test) ** 2)))
print('Root Mean Squared Error:', rmse)

# Predict the closing stock price for end_date + 1
last_year_data = data['Close'].values[-252:]
last_year_data_scaled = scaler.transform(last_year_data.reshape(-1, 1))
X = []
for i in range(time_steps, len(last_year_data_scaled)):
    X.append(last_year_data_scaled[i - time_steps:i, 0])

X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
predicted_price = model.predict(X)
predicted_price = scaler.inverse_transform(predicted_price)
print('Predicted closing stock price for March 31, 2023:', predicted_price[-1][0])