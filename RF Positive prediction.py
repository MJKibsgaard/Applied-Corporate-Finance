import yfinance as yf
import pandas as pd
import numpy as np
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

# Train a Random Forest model on the combined data of all 5 stocks
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the combined training set to check performance
train_predictions = model.predict(X_train)
mse = mean_squared_error(y_train, train_predictions)
print(f"Training MSE for all stocks: {mse}")



# Now we actually test the model on our validation data. We test if the stock picking actually provides value to the investment

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
    
    # Construct portfolio: invest in stock on days with positive predicted return
    test_data['Strategy_Return'] = test_data['Daily_Return'] * (test_data['Predicted_Return'] > 0)
    
    
    # Compute cumulative returns
    test_data['Cumulative_Strategy_Return'] = (1 + test_data['Strategy_Return']).cumprod() - 1
    test_data['Cumulative_Actual_Return'] = (1 + test_data['Daily_Return']).cumprod() - 1

    # sharp ratios
    strategy_sharpe_ratio = calculate_sharpe_ratio(test_data['Strategy_Return'])
    actual_sharpe_ratio = calculate_sharpe_ratio(test_data['Daily_Return'])
    
    test_data['Strategy_Sharpe_Ratio'] = strategy_sharpe_ratio
    test_data['Actual_Sharpe_Ratio'] = actual_sharpe_ratio
    

    return test_data[['Daily_Return', 'Predicted_Return', 'Strategy_Return', 
                      'Cumulative_Strategy_Return', 'Cumulative_Actual_Return',
                      'Strategy_Sharpe_Ratio', 'Actual_Sharpe_Ratio']]


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

        
# Display the results for one of the stocks to check, for example, META
print(backtest_results['META'])


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


     # Extract Sharpe ratios for each stock
    strategy_sharpe_ratios = [df['Strategy_Sharpe_Ratio'].iloc[0] for _, df in backtest_results.items()]
    actual_sharpe_ratios = [df['Actual_Sharpe_Ratio'].iloc[0] for _, df in backtest_results.items()]

    # Calculate average Sharpe ratios
    average_strategy_sharpe_ratio = np.mean(strategy_sharpe_ratios)
    average_actual_sharpe_ratio = np.mean(actual_sharpe_ratios)

    print(f"Average Model-Based Strategy Sharpe Ratio: {average_strategy_sharpe_ratio:.4f}")
    print(f"Average Holding Sharpe Ratio: {average_actual_sharpe_ratio:.4f}")


# Plot average performance across all stocks
plot_average_performance(backtest_results)











