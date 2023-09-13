import yfinance as yf
import pandas as pd
from Functions import calculate_daily_returns

# Liste over danske C25-virksomheder og deres tickers
c25_companies = [
    'MAERSK-A.CO',
    'NOVO-B.CO'  # Tilføjet Novo Nordisk
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
    
    # Tilføj volumen
    data['Volume'] = data['Volume']
    
    # Tilføj markedskapitalisering og P/E-forhold som konstante kolonner
    data['Market Cap'] = ticker_data.info['marketCap']
    data['P/E Ratio'] = ticker_data.info['trailingPE']
    
    # Tilføj aktiens navn som en overtitel
    stock_name = ticker_data.info['shortName']
    stock_header = pd.DataFrame({stock_name: ["" for _ in range(len(data))]})
    combined_data = pd.concat([stock_header, data], axis=1)
    
    # Sammenføj data til hoved-DataFrame
    c25_all_data = pd.concat([c25_all_data, combined_data], axis=0)

# Drop unødvendige kolonner og omarranger kolonner
columns_order = ['Daily Returns', 'Volume', 'Market Cap', 'P/E Ratio']
c25_all_data = c25_all_data[columns_order]

# Gem den konsoliderede data i Excel
c25_all_data.to_excel('consolidated_data_with_headers.xlsx', index=True)
