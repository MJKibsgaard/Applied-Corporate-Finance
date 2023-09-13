import yfinance as yf
import pandas as pd
from Functions import calculate_daily_returns

# Liste over danske C25-virksomheder og deres tickers
c25_companies = [
    'MAERSK-A.CO'
]

# Start- og slutdato for din dataindsamling
start_date = '2023-01-01'
end_date = '2023-09-01'

# DataFrame to store all data
c25_all_data = pd.DataFrame()

# Hent data for hver C25-virksomhed
for ticker in c25_companies:
    data = yf.download(ticker, start=start_date, end=end_date)
    ticker_data = yf.Ticker(ticker)
    
    # Calculate daily returns
    data['Daily Returns'] = data['Adj Close'].pct_change()
    
    # Add volume
    data['Volume'] = data['Volume']
    
    # Add market capitalization and P/E ratio as constant columns
    data['Market Cap'] = ticker_data.info['marketCap']
    data['P/E Ratio'] = ticker_data.info['trailingPE']
    
    # Concatenate data to the main DataFrame
    c25_all_data = pd.concat([c25_all_data, data], axis=0)

# Drop unwanted columns
c25_all_data = c25_all_data[['Daily Returns', 'Volume', 'Market Cap', 'P/E Ratio']]

# Save the consolidated data to Excel
c25_all_data.to_excel('consolidated_data.xlsx', index=True)
