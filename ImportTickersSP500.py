import pandas as pd
import numpy as np

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500_list = np.array(sp500[0]['Symbol'])

formatted_tickers = ', '.join(['"{}"'.format(ticker) for ticker in sp500_list])
print(formatted_tickers)

