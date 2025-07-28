import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries

# Set random seed for reproducibility
np.random.seed(42)

# 1. Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# 2. Preprocess Data
def preprocess_data(data, time_steps=60, split_ratio=0.8):
    """
    Prepare data for LSTM:
    - Normalize data
    - Create sequences
    - Split into train/test sets
    """
    # Use only Close price
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create sequences
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

# 3. Build LSTM Model
def build_lstm_model(input_shape):
    """
    Create LSTM model architecture
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# 4. Plot Results
def plot_results(actual, predicted, ticker):
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(14, 6))
    plt.plot(actual, color='blue', label=f'Actual {ticker} Price')
    plt.plot(predicted, color='red', label=f'Predicted {ticker} Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Program
def main():
    # Configuration
    TICKER = 'AAPL'  # Stock symbol
    TIME_STEPS = 60   # Number of time steps to look back
    EPOCHS = 100      # Training epochs
    BATCH_SIZE = 32   # Batch size
    
    # Date range (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # 1. Fetch data
    stock_data = fetch_stock_data(TICKER, start_date, end_date)
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data, TIME_STEPS)
    
    # 3. Build model
    model = build_lstm_model((X_train.shape[1], 1))
    print(model.summary())
    
    # 4. Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 5. Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 6. Plot results
    plot_results(y_test_actual, predictions, TICKER)
    
    # 7. Predict next day price
    last_sequence = X_test[-1].reshape(1, TIME_STEPS, 1)
    next_day_pred = model.predict(last_sequence)
    next_day_pred = scaler.inverse_transform(next_day_pred)
    print(f"\nPredicted {TICKER} price for next trading day: ${next_day_pred[0][0]:.2f}")

    API_KEY = "9D82LNFKY3Y4FDOI"
ts = TimeSeries(key=API_KEY, output_format='pandas')

data, meta_data = ts.get_intraday(symbol='AAPL', interval='1min', outputsize='compact')

# Print the latest close price
latest_time = data.index[0]
latest_price = data['4. close'][0]

print(f"Latest AAPL Price at {latest_time}: ${latest_price}")

if __name__ == "__main__":
    main()