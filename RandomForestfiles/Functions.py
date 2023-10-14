
#Function 1 - Getting the data from the web
import yfinance as yf
import pandas as pd

def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical stock data for given tickers from Yahoo Finance.

    Parameters:
    - tickers (list): List of stock ticker symbols.
    - start_date (str): Start date in format 'YYYY-MM-DD'.
    - end_date (str): End date in format 'YYYY-MM-DD'.

    Returns:
    - dict: A dictionary where each key is a ticker and its associated value is a DataFrame containing the stock's historical data for the given date range.
    """
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return stock_data



#Function 2 - 20 day moving average
def moving_average(data, window=20):
    return data['Close'].rolling(window=window).mean()


#Function 3 - RSI
def rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


#Function 4 - MACD
def macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    MACD_line = short_ema - long_ema
    signal_line = MACD_line.ewm(span=signal_window, adjust=False).mean()
    
    return MACD_line, signal_line


#Function 5 - Lagged return
def lagged_return(data, days=1):
    return data['Close'].pct_change(periods=days).shift(days)


#Function 6 - Bollinger bands
def bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, lower_band


#Function 7 - PE Ratio
def pe_ratio(data, earnings_per_share):
    return data['Close'] / earnings_per_share


#Function 8 - Daily return
def daily_returns(data):
    daily_returns = data.pct_change().dropna()
    return daily_returns

