### Extracting data from yahoo finance. It should be easily scalable. We can just add tickers to expand the model
import yfinance as yf
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from Functions import fetch_data_with_fundamentals, compute_technical_indicators, compute_lagged_returns, split_data, calculate_sharpe_ratio

# Setting parameters for fetching data
tickers = ["AAPL", "MSFT", "META"]
start_date = '2015-01-01'
end_date = '2020-01-01'

# Calling the function that gets the data
stock_data = fetch_data_with_fundamentals(tickers, start_date, end_date)

# Print the data for one of the stocks to check
print(stock_data['META'].tail())

# Compute technical indicators for each stock
for ticker, df in stock_data.items():
    stock_data[ticker] = compute_technical_indicators(df)


# Compute lagged returns for each stock
for ticker, df in stock_data.items():
    stock_data[ticker] = compute_lagged_returns(df)

# Print the data for one of the stocks to check
print(stock_data['META'].tail(10))


# Split data for each stock in the stock_data dictionary
train_data = {}
test_data = {}

for ticker, df in stock_data.items():
    train, test = split_data(df)
    train_data[ticker] = train
    test_data[ticker] = test

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create the target column for each stock and concatenate to form a combined dataset
all_train_data = pd.concat([df.assign(Ticker=ticker) for ticker, df in train_data.items()])
all_train_data['Target'] = all_train_data.groupby('Ticker')['Daily_Return'].shift(-1)
all_train_data.dropna(inplace=True)  # Drop rows with NaN targets

# Define features and target
features = ['MA20', 'RSI', 'PE_Ratio', 'Market_Cap'] + [f'Lagged_Return_{i}' for i in range(1, 6)]
target = 'Target'

# Split combined dataset into X (features) and y (target)
X_train = all_train_data[features]
y_train = all_train_data[target]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define and train the MLP model
model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), 
                     activation='relu', 
                     solver='adam', 
                     max_iter=500, 
                     random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the training set to check performance
train_predictions = model.predict(X_train_scaled)
mse = mean_squared_error(y_train, train_predictions)
print(f"Training MSE for all stocks with MLP: {mse}")

