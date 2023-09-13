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

# Hent data for hver C25-virksomhed
c25_data = yf.download(c25_companies, start=start_date, end=end_date)

# Kald funktionen med C25-virksomhederne
c25_daily_returns = calculate_daily_returns(c25_companies, start_date, end_date)

# Opret en kolonne "Ticker" baseret på indeksnavnet
c25_daily_returns['Ticker'] = c25_daily_returns.index

# Hent andre data
pe_ratios = {}
market_caps = {}

for ticker in c25_companies:
    info = yf.Ticker(ticker).info
    pe_ratios[ticker] = info.get('trailingPE')
    market_caps[ticker] = info.get('marketCap')

# Opret DataFrames for P/E-forhold og Market Capitalization
pe_df = pd.DataFrame(list(pe_ratios.items()), columns=['Ticker', 'PE Ratio'])
market_cap_df = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'Market Cap'])

# Sæt "Ticker" som indeks i P/E-forhold og Market Capitalization DataFrames
pe_df.set_index('Ticker', inplace=True)
market_cap_df.set_index('Ticker', inplace=True)

# Flet data om P/E-forhold og Market Capitalization med c25_daily_returns
c25_daily_returns = pd.concat([c25_daily_returns, pe_df, market_cap_df], axis=1, join='inner')

# Print nogle data fra funktionen
print(c25_daily_returns.head())

# Fjern den midlertidige "Ticker" kolonne
c25_daily_returns.drop(columns=['Ticker'], inplace=True)

# Gem daglige afkast inklusive ekstra data i en Excel-fil
c25_daily_returns.to_excel('daily_returns_with_data.xlsx', index=True)
