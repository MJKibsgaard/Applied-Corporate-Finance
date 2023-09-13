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

# Opret en DataFrame for at gemme alle data
c25_data = pd.DataFrame()

# Hent data for hver C25-virksomhed
for ticker in c25_companies:
    data = yf.download(ticker, start=start_date, end=end_date)
    ticker_data = yf.Ticker(ticker)
    
    # Tilføj 'Volume' til data
    data['Volume'] = data['Volume']
    
    # Tilføj markedskapitalisering og P/E-forhold som konstante kolonner
    data['Market Cap'] = ticker_data.info['marketCap']
    data['P/E Ratio'] = ticker_data.info['trailingPE']
    
    # Tilføj data til den samlede DataFrame
    c25_data = pd.concat([c25_data, data], axis=0)

# Vis de første rækker af data for alle virksomheder
print(c25_data.head())

# Kald funktionen med C25-virksomhederne
c25_daily_returns = calculate_daily_returns(c25_companies, start_date, end_date)

# Print nogle data fra funktionen
print(c25_daily_returns.head())

# Gem daglige afkast i en Excel-fil
c25_daily_returns.to_excel('daily_returns.xlsx', index=True)
