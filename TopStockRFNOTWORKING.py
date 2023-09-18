### Extracting data from yahoo finance. It should be easily scalable. We can just add tickers to expand the model
import yfinance as yf
from tqdm import tqdm
import numpy as np

def fetch_data_with_fundamentals(tickers, start_date, end_date):
    """
    Fetch daily stock data along with PE ratio and market cap for given tickers from Yahoo Finance.

    Parameters:
    - tickers (list): List of stock ticker symbols.
    - start_date (str): Start date in format 'YYYY-MM-DD'.
    - end_date (str): End date in format 'YYYY-MM-DD'.

    Returns:
    - data (dict): Dictionary with ticker symbols as keys and corresponding data as values.
    """
    data = {}
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            # Fetch PE ratio and market cap
            ticker_obj = yf.Ticker(ticker)
            pe_ratio = ticker_obj.info.get('trailingPE', None)
            market_cap = ticker_obj.info.get('marketCap', None)
            
            # Only if both stock_data, PE ratio, and market cap are available, store them in the dictionary
            if stock_data is not None and pe_ratio is not None and market_cap is not None:
                stock_data['PE_Ratio'] = pe_ratio
                stock_data['Market_Cap'] = market_cap
                data[ticker] = stock_data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue
    return data


# Pick the data we want to use
tickers = ["META", "MMM", "AOS", "ABT", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", "AAPL", "AMAT", "APTV", "ACGL", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BBWI", "BAX", "BDX", "WRB", "BRK.B", "ZTS","CTSH", "CL", "CMCSA", "CMA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CTVA", 
"CSGP", "COST", "CTRA", "CCI", "CSX", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL"]
start_date = '2015-01-01'
end_date = '2020-01-01'
stock_data = fetch_data_with_fundamentals(tickers, start_date, end_date)

# Print the data for one of the stocks to check
print(stock_data['META'].tail())


### Next, we calculate some technical indicators

import pandas as pd

def compute_technical_indicators(data):
    """
    Compute 20-day moving average and RSI for the stock data.

    Parameters:
    - data (DataFrame): Stock data.

    Returns:
    - data (DataFrame): Stock data with added MA20 and RSI columns.
    """
    # 20-day Moving Average
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    # Compute Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# Compute technical indicators for each stock
for ticker, df in stock_data.items():
    stock_data[ticker] = compute_technical_indicators(df)

# Print the data for one of the stocks to check
print(stock_data['META'].tail())



def compute_lagged_returns(data, lag_days=5):
    """
    Compute lagged returns for the stock data.

    Parameters:
    - data (DataFrame): Stock data with Close prices.
    - lag_days (int): Number of lagged days to compute.

    Returns:
    - data (DataFrame): Stock data with added lagged return columns.
    """
    # Compute daily returns
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Create lagged return columns
    for i in range(1, lag_days + 1):
        data[f'Lagged_Return_{i}'] = data['Daily_Return'].shift(i)
    
    return data

# Compute lagged returns for each stock
for ticker, df in stock_data.items():
    stock_data[ticker] = compute_lagged_returns(df)

# Print the data for one of the stocks to check
print(stock_data['META'].tail(10))


# Assuming you already have the stock_data dictionary populated
meta_data = stock_data['META']

# Save the data for META to an Excel file
file_path = "META_stock_data.xlsx"
meta_data.to_excel(file_path)

print(f"Data saved to {file_path}")




def split_data(data, train_fraction=0.8):
    """
    Split time series data into training and testing sets.

    Parameters:
    - data (DataFrame): Stock data.
    - train_fraction (float): Fraction of data to be used for training.

    Returns:
    - train (DataFrame): Training data.
    - test (DataFrame): Testing data.
    """
    train_size = int(len(data) * train_fraction)
    train = data[:train_size]
    test = data[train_size:]
    return train, test

# Split data for each stock in the stock_data dictionary
train_data = {}
test_data = {}

for ticker, df in stock_data.items():
    train, test = split_data(df)
    train_data[ticker] = train
    test_data[ticker] = test

# Print the shapes of the training and testing data for META to check
print(f"META - Training data shape: {train_data['META'].shape}")
print(f"META - Testing data shape: {test_data['META'].shape}")



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

# Train a Random Forest model on the combined data of all 5 stocks
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the combined training set to check performance
train_predictions = model.predict(X_train)
mse = mean_squared_error(y_train, train_predictions)
print(f"Training MSE for all stocks: {mse}")



def backtest_model(test_data, model, features):
    """
    Backtest the trained model on the testing data.

    Parameters:
    - test_data (DataFrame): Testing data.
    - model (Regressor): Trained regression model.
    - features (list): List of feature columns to use in prediction.

    Returns:
    - results (DataFrame): DataFrame with actual and predicted returns, and the strategy's performance.
    """
    # Predict returns
    test_data['Predicted_Return'] = model.predict(test_data[features])
    
    # Initialize the Top10_Strategy_Return column with NaNs
    test_data['Top10_Strategy_Return'] = np.nan
    
    # For each day, pick the top 10 stocks and set the average return of these stocks as the strategy return for that day
    for date, group in test_data.groupby('Date'):
        top_10 = group.sort_values(by='Predicted_Return', ascending=False).head(10)
        avg_return = top_10['Daily_Return'].mean()
        test_data.loc[test_data['Date'] == date, 'Top10_Strategy_Return'] = avg_return

    # Compute cumulative returns
    test_data['Cumulative_Top10_Strategy_Return'] = (1 + test_data['Top10_Strategy_Return']).cumprod() - 1
    test_data['Cumulative_Actual_Return'] = (1 + test_data['Daily_Return']).cumprod() - 1
    
    return test_data[['Date', 'Daily_Return', 'Predicted_Return', 'Top10_Strategy_Return', 
                      'Cumulative_Top10_Strategy_Return', 'Cumulative_Actual_Return']]



## Backtest the model on the testing data for all stocks
backtest_results = {}
for ticker in test_data.keys():
    if len(test_data[ticker]) == 0:
        print(f"No test data available for {ticker}. Skipping...")
        continue
    if set(features).issubset(test_data[ticker].columns):
        backtest_results[ticker] = backtest_model(test_data[ticker], model, features)
    else:
        print(f"Required features not available for {ticker}. Skipping...")





import matplotlib.pyplot as plt

def plot_average_performance(backtest_results):
    """
    Plot the average cumulative returns of the model-based strategy vs. holding across all stocks.

    Parameters:
    - backtest_results (dict): Dictionary with tickers as keys and backtesting results as values.
    """
    # Initialize a list to store the 'Cumulative_Strategy_Return' columns for all stocks
    cumulative_returns = []
    cumulative_returns_actual = []

    # Extract the 'Cumulative_Strategy_Return' column for each stock and append to the list
    for _, df in backtest_results.items():
        cumulative_returns.append(df['Cumulative_Strategy_Return'])

    for _, df in backtest_results.items():
        cumulative_returns_actual.append(df['Cumulative_Actual_Return'])

    # Assuming you're using pandas, you can use the concat function to concatenate these columns and then compute the mean along the horizontal axis (axis=1) to get the daily average
    average_strategy_returns = pd.concat(cumulative_returns, axis=1).mean(axis=1)
    average_actual_returns = pd.concat(cumulative_returns_actual, axis=1).mean(axis=1)


    plt.figure(figsize=(14, 7))
    plt.plot(average_strategy_returns, label='Average Model-Based Strategy', color='blue')
    plt.plot(average_actual_returns, label='Average Holding', color='orange')
    plt.title('Average Cumulative Returns Across All Stocks')
    plt.xlabel('Date')
    plt.ylabel('Average Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot average performance across all stocks
plot_average_performance(backtest_results)


