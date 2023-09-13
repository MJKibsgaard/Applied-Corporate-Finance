import yfinance as yf
import pandas as pd
import datetime

# Define the ticker symbol
tickerSymbol = 'TSLA'

# Get data for this ticker
tickerData = yf.Ticker(tickerSymbol)

# Fetch the historical data for the last year
tesla_data = tickerData.history(period='1y')

# Extracting the required columns: 'Close' for price, 'Volume', and 'Market Cap'
tesla_data = tesla_data[['Close', 'Volume']]

# Note: yfinance doesn't provide market cap on a daily basis. Instead, you can get the current market cap.
current_market_cap = tickerData.info['marketCap']

print(tesla_data.head())  # Display the first few rows
