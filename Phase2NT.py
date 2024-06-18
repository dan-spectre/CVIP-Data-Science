import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('monthly_data.csv')
data['DATE'] = pd.to_datetime(data['DATE'])  

# Converting numeric columns to appropriate data types
numeric_cols = ['MonthlyMeanTemperature', 'MonthlyTotalLiquidPrecipitation']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna() 
print(data.describe())

# Temperature Trend Over Time
plt.figure(figsize=(14, 7))
plt.plot(data['DATE'], data['MonthlyMeanTemperature'], color='blue', label='Monthly Mean Temperature')
plt.title('Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.show()

# Precipitation Trend Over Time
plt.figure(figsize=(14, 7))
plt.plot(data['DATE'], data['MonthlyTotalLiquidPrecipitation'], color='green', label='Monthly Total Precipitation')
plt.title('Precipitation Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.grid(True)
plt.legend()
plt.show()

# Monthly Average Temperature
monthly_avg_temp = data.groupby(data['DATE'].dt.to_period('M'))['MonthlyMeanTemperature'].mean()
plt.figure(figsize=(14, 7))
monthly_avg_temp.plot(kind='line', color='blue')
plt.title('Monthly Average Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# Seasonal Analysis
data['Month'] = data['DATE'].dt.month
data['Year'] = data['DATE'].dt.year

# Monthly Mean Temperature
plt.figure(figsize=(14, 7))
sns.boxplot(x='Month', y='MonthlyMeanTemperature', data=data)
plt.title('Monthly Mean Temperature Distribution')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# Save cleaned data for further analysis
data.to_csv('cleaned_monthly_data.csv', index=False)
