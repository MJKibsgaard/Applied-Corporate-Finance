import yfinance as yf
import pandas as pd


def calculate_daily_returns(tickers, start, end):
    # Opret en tom DataFrame til at gemme daglige afkast
    daily_returns = pd.DataFrame()

    # Hent data for hver aktie og beregn daglige afkast
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)
        adj_close = data['Adj Close']
        daily_returns[ticker] = adj_close.pct_change()

    return daily_returns





