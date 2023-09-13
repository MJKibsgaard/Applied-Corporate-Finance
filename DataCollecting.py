import yfinance as yf
import pandas as pd
from Functions import calculate_daily_returns

# Liste over danske C25-virksomheder og deres tickers
c25_companies = [
    'MAERSK-A.CO',
    'NOVO-B.CO'
]

# Start- og slutdato for din dataindsamling
start_date = '2023-01-01'
end_date = '2023-09-01'

# DataFrame til at gemme alle data
c25_all_data = pd.DataFrame()

# Hent data for hver C25-virksomhed
for ticker in c25_companies:
    data = yf.download(ticker, start=start_date, end=end_date)
    ticker_data = yf.Ticker(ticker)
    
    # Beregn daglige afkast
    data['Daily Returns'] = data['Adj Close'].pct_change()
    
    # Tilføj markedskapitalisering og P/E-forhold som konstante kolonner
    data['Market Cap'] = ticker_data.info['marketCap']
    data['P/E Ratio'] = ticker_data.info['trailingPE']
    
    # Omdøb kolonner for at have aktiens navn som et præfiks
    stock_name = ticker_data.info['shortName']
    data.columns = [f"{stock_name} {col}" for col in data.columns]
    
    # Sammenføj data til hoved-DataFrame
    c25_all_data = pd.concat([c25_all_data, data], axis=1)

# Drop unødvendige kolonner og omarranger kolonner
c25_all_data = c25_all_data.dropna(axis=1, how='all')

# Gem den konsoliderede data i Excel
c25_all_data.to_excel('consolidated_data.xlsx', index=True)
