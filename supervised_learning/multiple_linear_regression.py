'''
Multiple Linear Regression: Involves more than one independent variable (feature).
The model is represented as a hyperplane in higher dimensions.

Equation:
y=β0+β1⋅X1+β2⋅X2+⋯+βn⋅Xn+ϵ
Where:

    y is the dependent variable (target),
    X1,X2,…,Xn​ are independent variables (features),
    β0​ is the intercept,
    β1,β2,…,βn are the coefficients of each feature,
    ϵϵ is the error term.
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from utils import find_outliers, fix_outliers


df = pd.read_csv("student_mat.csv", delimiter=';')

# Data Cleaning
if df.isna().any().any():
    print("Fixing null values")
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].median(), inplace=True)
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

# Encode category data to numerical representation to apply regression
for column in df.select_dtypes(include=["category", "object"]):
    df[column] = LabelEncoder().fit_transform(df[column])

# Find and fix outliers
for column in df.select_dtypes(include=[np.number]).columns:
    fix_outliers(df, column)


# Split the data for target and features
y = df[['G1', 'G2', 'G3']]
# Using single target as multi variate target requires a more complex algorithm
X = df.drop(columns=['G3'])

# Standardize the numerical features to reduce the dominance of large values
# This process transforms each feature (column) to have a mean of 0 and a 
# standard deviation of 1. It ensures that each feature contributes equally 
# to the machine learning model.
# Remember to stardize only the non target values.
scaler = StandardScaler()
X_standard = scaler.fit_transform(X) # This will generate a numpy array
# Convert back to pandas dataframe
X = pd.DataFrame(X, columns=X.columns)

# Split the data in training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the predictions to calculate the model accuracy on different metrics
y_pred = model.predict(X_test)

# Calculate all the metrics and print
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) # How well the model is a fit

print(f"Mean absolute error: {mae}")
print(f"Mean squared error: {mse}")
print(f"R2 Score: {r2}")

'''Output
Mean absolute error: 0.5087610787809732
Mean squared error: 1.6116191208599588
R2 Score: 0.9214037841614807
'''

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Scatter Plot
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color='r', linestyle='--')
plt.title("Scatter Plot")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()