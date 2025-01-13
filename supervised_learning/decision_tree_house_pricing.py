'''
An optimal Decision Tree:
Has pure or nearly pure leaf nodes to ensure high accuracy.
Maintains sufficiently large leaf nodes to prevent overfitting by basing decisions on general patterns rather than noise.
Controls the depth of the tree to avoid overly complex structures, using techniques like pruning and parameter tuning.
Balances the tree growth to ensure that all paths are appropriately generalized without excessive complexity.

Predicts the house prices on basis of provided features using california house pricing dataset.
'''

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

from utils.utils import fix_outliers


df = fetch_california_housing(as_frame=True).frame

if df.isna().any().any():
    for column in df.select_dtypes(include=[np.number]):
        df[column].fillna(df[column].median(), inplace=True)
    for column in df.select_dtypes(include=['object', 'category']):
        df[column].fillna(df[column].mode()[0], inplace=True)

# Seperate the target and features
y = df['MedHouseVal']
X = df.drop(columns=['MedHouseVal'])

# Fix the outliers
for column in df.select_dtypes(include=[np.number]):
    df[column] = fix_outliers(df, column)

# Standardization
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

X = pd.DataFrame(X_s)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.3, random_state=42
)

# Initialize the model
model = DecisionTreeRegressor(random_state=42)

# Tune the hyperparameters using gridsearchcv
param_grid = {
    "max_depth": [3, 7, 9, None],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ['sqrt', 'log2', None],
    "criterion": ['squared_error', 'absolute_error']
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)
grid_search.fit(X_train, y_train)

regressor = grid_search.best_estimator_
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Metrics calculation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R2 score: {r2}")

'''
Output:-
Mean Squared Error: 0.4226940204596524
Mean Absolute Error: 0.443422544961073
R2 score: 0.6695356097548433
'''
