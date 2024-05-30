import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('transact_data.csv')

# Display the first few rows of the dataframe
print("Data:")
print(data.head())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Convert the DAY column to datetime format
data['DAY'] = pd.to_datetime(data['DAY'], format='%d', errors='coerce')
data['MONTH'] = pd.to_datetime(data['MONTH'], format='%m', errors='coerce')

# Fill missing values with 0 (assuming missing values mean no sales)
data.fillna(0, inplace=True)

# Check for duplicates and remove them
data.drop_duplicates(inplace=True)

# Derive the date from DAY and MONTH
data['DATE'] = data['DAY'].dt.strftime('2021-%m-%d')

# Get unique store codes
stores = data['STORECODE'].unique()

# Top 10 spending customers
customer_spending = data.groupby('BILL_ID')['BILL_AMT'].sum()
top_spending_customers = customer_spending.nlargest(10)
print("Top 10 Spending Customers:")
print(top_spending_customers)

# Highest, lowest, and average bill
bills = data['BILL_AMT']
highest_bill = bills.max()
lowest_bill = bills.min()
average_bill = bills.mean()

print(f"\nHighest bill: {highest_bill}")
print(f"Lowest bill: {lowest_bill}")
print(f"Average bill: {average_bill}")

# Analyze popular products
popular_products = data['BRD'].value_counts().nlargest(10)
print("\nTop 10 Popular Products:")
print(popular_products)

# Plot overall transaction trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='DAY', y='BILL_AMT')
plt.title('Overall Transaction Trends')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.grid(True)
plt.show()

# Plot transaction trends per store
plt.figure(figsize=(12, 6))
for store in stores:
    store_data = data[data['STORECODE'] == store]
    plt.plot(store_data['DATE'], store_data['BILL_AMT'], label=f'Store {store}')

plt.title('Transaction Trends per Store')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize transaction distribution by month
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='MONTH', y='BILL_AMT')
plt.title('Transaction Distribution by Month')
plt.xlabel('Month')
plt.ylabel('BILL_AMT')
plt.show()

# Analyze purchase frequency by DAY of each month
plt.figure(figsize=(12, 6))
purchase_frequency = data.groupby(['MONTH', 'DAY']).size().reset_index(name='Frequency')
sns.barplot(data=purchase_frequency, x='DAY', y='Frequency', hue='MONTH')
plt.title('Purchase Frequency by Day of Each Month')
plt.xlabel('Day of Week')
plt.ylabel('Frequency')
plt.show()


