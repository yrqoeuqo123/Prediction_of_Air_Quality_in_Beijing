import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
file_names = [
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "PRSA_Data_Changping_20130301-20170228.csv",
    "PRSA_Data_Dingling_20130301-20170228.csv",
    "PRSA_Data_Dongsi_20130301-20170228.csv",
    "PRSA_Data_Guanyuan_20130301-20170228.csv",
    "PRSA_Data_Gucheng_20130301-20170228.csv",
    "PRSA_Data_Huairou_20130301-20170228.csv",
    "PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "PRSA_Data_Shunyi_20130301-20170228.csv",
    "PRSA_Data_Tiantan_20130301-20170228.csv",
    "PRSA_Data_Wanliu_20130301-20170228.csv",
    "PRSA_Data_Wanshouxigong_20130301-20170228.csv"
]
season_mapping = {
    1: 'Winter', 2: 'Winter', 3: 'Spring',
    4: 'Spring', 5: 'Spring', 6: 'Summer',
    7: 'Summer', 8: 'Summer', 9: 'Fall',
    10: 'Fall', 11: 'Fall', 12: 'Winter'
}


df = pd.read_csv(
    "PRSA_Data_Changping_20130301-20170228.csv", encoding='ISO-8859-1')

df = df[['year', 'month', 'day', 'hour', 'PM2.5', 'PM10']]
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

# Drop unnecessary columns
df = df.drop(['day', 'hour'], axis=1)

# Group by 'year' and 'month' and calculate the mean for PM2.5
monthly_average_pm25 = df.groupby(
    [df['datetime'].dt.year, df['datetime'].dt.month])['PM10'].mean()

# Plotting the graph
plt.figure(figsize=(12, 6))

plt.plot(monthly_average_pm25.index.map(
    lambda x: f"{x[0]}-{x[1]:02d}"), monthly_average_pm25.values, marker='o', label='Average PM10')

# Add a red horizontal line at value 55
plt.axhline(y=50, color='red', linestyle='--', label='Threshold for Unhealthy')

plt.title('Average PM10 Level for Each Month of the Year')
plt.xlabel('Year-Month')
plt.ylabel('Average PM10 Level')
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
