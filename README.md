Apple (AAPL) Stock Price Prediction
This project aims to predict the closing prices of Apple Inc. (AAPL) stock using an LSTM (Long Short-Term Memory) model. The project involves downloading stock data, preprocessing the data, training an LSTM model, and visualizing the prediction results.

1. Importing Necessary Libraries
The following Python libraries are used in this project:

python

import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
import yfinance as yf
2. Downloading Stock Data
The Yahoo Finance (yfinance) library is used to download AAPL stock data from January 1, 2010, to January 1, 2024:

python

ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2024-01-01'
stock_data = yf.download(ticker, start=start_date, end=end_date)
3. Visualizing Closing Prices
The closing prices over time are visualized using matplotlib:

python

plt.style.use('dark_background')
plt.figure(figsize=(16, 8))
plt.title('Close')
plt.plot(stock_data['Close'])
plt.xlabel('History', fontsize=18)
plt.ylabel('Close (USD)', fontsize=18)
plt.show()
4. Data Preparation
Closing prices are filtered and scaled:

python

data = stock_data.filter(['Close'])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
The training data is set to 80% of the total dataset:

python

training_data_len = int(len(dataset) * 0.8)
train_data = scaled_data[:training_data_len, :]
test_data = scaled_data[training_data_len - 60:, :]
5. Creating Training and Testing Datasets
Training and testing datasets are created:

python

x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
6. Building and Training the LSTM Model
An LSTM model is built and trained:

python

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stopping])
7. Testing the Model and Making Predictions
Test data is prepared and model predictions are made:

python

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
8. Visualizing Prediction Results
The prediction results are visualized along with training and testing data:

python

train = data[:training_data_len]
valid = data[training_data_len:]
valid['predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Model Forecast Results – AAPL Stock')
plt.xlabel('History', fontsize=18)
plt.ylabel('Closing Price (USD)', fontsize=18)

plt.plot(train['Close'], color='red', label='Training Data Set')
plt.plot(valid['Close'], color='yellow', label='Validation Data Set')
plt.plot(valid['predictions'], color='blue', label='Predictions')

plt.legend(loc='upper left')
plt.show()
9. Analyzing the Last 20 Days
The predicted closing prices and the actual closing prices for the last 20 days are analyzed:

python

print(valid.tail(20))
10. Predicting Price One Month Ahead
Using the last 60 days of data, a prediction for the next month's price is made:

python

last_60_days = scaled_data[-60:]
X_test = np.array([last_60_days])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price_1months = model.predict(X_test)
pred_price_1months = scaler.inverse_transform(pred_price_1months)
print(f'{ticker} Price after 1 month for: {pred_price_1months[0][0]}')
Summary
In this project, an LSTM model was used to predict the closing prices of Apple Inc. (AAPL) stock. The model was trained on closing prices from 2010 to 2024. The model's performance was evaluated on test data, and a forecast for the next month's closing price was made. Techniques such as early stopping were used to enhance model performance. The results and predictions were visualized and analyzed to understand the model's accuracy and performance. This approach demonstrates how machine learning models can be applied to financial data for time series forecasting.
