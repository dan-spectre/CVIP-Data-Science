import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset (assuming it's in CSV format)
data = pd.read_csv('portfolio_data.csv', parse_dates=['Date'], index_col='Date') #have to import csv file data to run this program

# Get unique stock symbols
stocks = data.columns

# Plot original time series, trend, seasonal, and residual for each stock
for stock in stocks:
    # Decompose time series
    decomposition = seasonal_decompose(data[stock], model='additive', period=252)  # Assuming 252 trading days in a year

    # Plot original time series
    plt.figure(figsize=(12, 10))

    # Original time series
    plt.subplot(4, 1, 1)
    plt.plot(data[stock], label='Original')
    plt.legend(loc='upper left')
    plt.title(f'{stock} Stock Prices')

    # Trend
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label='Trend', color='green')
    plt.legend(loc='upper left')
    plt.title(f'{stock} Trend')

    # Seasonal
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label='Seasonal', color='red')
    plt.legend(loc='upper left')
    plt.title(f'{stock} Seasonal')

    # Residual
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label='Residual', color='purple')
    plt.legend(loc='upper left')
    plt.title(f'{stock} Residual')

    plt.tight_layout()
    plt.show()

# Plot stock prices over time for each stock
plt.figure(figsize=(12, 6))
for stock in stocks:
    plt.plot(data.index, data[stock], label=stock)
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
