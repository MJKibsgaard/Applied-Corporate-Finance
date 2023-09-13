import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt  # Import this for showing the plot

##### THIS PART COLLECTS AND PREPARES THE DATA. WE ONLY USE MICROSOFT FOR THE TESTING #####

# We collect data for microsoft
msft = yf.Ticker("MSFT")
msft_hist = msft.history(period="max")

DATA_PATH = "msft_data.json"

if os.path.exists(DATA_PATH):
    # Read from file if we've already downloaded the data.
    with open(DATA_PATH) as f:
        msft_hist = pd.read_json(f)  # Read from file object, not string filename
else:
    msft = yf.Ticker("MSFT")
    msft_hist = msft.history(period="max")

    # Save file to json in case we need it later. This prevents us from having to re-download it every time.
    msft_hist.to_json(DATA_PATH)

# We take a look at the data
print(msft_hist.head(5))

# Visualize microsoft stock prices
msft_hist.plot.line(y="Close", use_index=True)

# Show the plot
plt.show()


##### NOW WE SET UP THE TARGET. WE PICK THE DATA OUR MODEL SHOULD PREDICT FROM #####
# Ensure we know the actual closing price
data = msft_hist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})

# Setup our target.  This identifies if the price went up or down
data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

# Show the data for data
data.head()

# Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)

# Show data for microsoft
msft_prev.head()


# Create our training data
predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(msft_prev[predictors]).iloc[1:]

data.head(5)

