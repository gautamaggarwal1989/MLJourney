'''
This script utilizes SBI bank historical data to train a model to
predict price of the share next day. We will be using RandomForestAggressor
to make the model more accurate.
This is our first approach toward this problem. We might try different other models
or different type of trainings for making it more accurate.
We will be using feature engineering to add more features in the dataset and make predictions
on the basis of it.
datasource:-
https://in.investing.com/equities
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.utils import fix_outliers

df = pd.read_csv("datasets/sbi_historical.csv")

# Fill missing data with mean and modes
if df.isna().any().any(): 
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].mean(), inplace=True)
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

# Feature Engineering
df['prev_price'] = df['Price'].shift(1)
df['prev_open'] = df['Open'].shift(1)
df['prev_high'] = df['High'].shift(1)
df['prev_low'] = df['Low'].shift(1)

# Moving averages
df['ma_5'] = df['Price'].rolling(window=5).mean()
df['ma_10'] = df['Price'].rolling(window=10).mean()

# Calculate the dispersion of data from mean: volatility
df["vol_5"] = df['Price'].rolling(window=5).std()
df["vol_10"] = df['Price'].rolling(window=10).std()

# Momentum indicator
df['mom_5'] = df['Price'] - df['Price'].shift(5)

# Cycling features
df['day_of_week'] = pd.to_datetime(df['Date'], dayfirst=True).dt.day_of_week
df['week'] =  pd.to_datetime(df['Date'], dayfirst=True).dt.isocalendar().week
df['month'] = pd.to_datetime(df['Date'], dayfirst=True).dt.month
df['day_of_year'] = pd.to_datetime(df['Date'], dayfirst=True).dt.day_of_year

features = ['prev_price', 'prev_open', 'prev_high', 'prev_low', 'ma_5', 'ma_10',
            'vol_5', 'vol_10', 'mom_5', 'day_of_week', 'day_of_year', 'week', 'month']

# Split the features and target price
X = df[features]
y = df['Price']

# Standardize the dataframe
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

X = pd.DataFrame(X_s)

# Split the training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Instantiate the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# collect the prediction
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Square Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R2 Score: {r2}')

# Predict today price
latest_data = df.iloc[-1]

# Create prediction parameters for today
today_test_data = pd.DataFrame(
    {
    'prev_price': [latest_data['Price']],
    'prev_open': [latest_data['Open']],
    'prev_high': [latest_data['High']],
    'prev_low': [latest_data['Low']],
    'ma_5': [df['Price'].rolling(window=5).mean().iloc[-1]],
    'ma_10':  [df['Price'].rolling(window=10).mean().iloc[-1]],
    'vol_5': [df['Price'].rolling(window=5).std().iloc[-1]],
    'vol_10': [df['Price'].rolling(window=10).std().iloc[-1]],
    'mom_5': [df['Price'].iloc[-1] - df['Price'].shift(5).iloc[-1]],
    'day_of_week': [pd.Timestamp('today').day_of_week],
    'day_of_year': [pd.Timestamp('today').day_of_year],
    'week': [pd.Timestamp('today').week],
    'month': [pd.Timestamp('today').month]
    }
)

today_data_scaled = scaler.transform(today_test_data)

today_prediction = model.predict(today_data_scaled)

print(f"Today's Prediction: {today_prediction}")

'''
Output:-
Mean Square Error: 16.306501368000006
Mean Absolute Error: 2.3144120000000084
R2 Score: 0.9994998607520263
Today's Prediction: [310.7515] # This is highly wrong :)
'''