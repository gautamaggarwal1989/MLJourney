'''
Linear regression is a statistical method used for modeling the relationship between a dependent variable
(target) and one or more independent variables (features). The goal of linear regression is to find the 
best-fitting line (or hyperplane in multiple dimensions) that minimizes the difference between the actual 
and predicted values.

Simple Linear Regression: Involves one independent variable (feature) and one dependent variable (target). The model is represented by a straight line.

Equation:
y=β0+β1⋅X+ϵ
Where:

    y is the dependent variable (target),
    X is the independent variable (feature),
    β0​ is the intercept (constant),
    β1​ is the slope (coefficient of XX),
    ϵ is the error term (residual).

Assumptions of Linear Regression

Linear regression relies on several assumptions for the model to be valid and produce reliable results:

a. Linearity:
The relationship between the independent variables and the dependent variable must be linear. In simple terms, the dependent variable should be a straight-line function of the independent variable(s).

b. Independence of Errors:
The residuals (the difference between observed and predicted values) should be independent. This means that the errors at one observation should not be correlated with the errors at another observation.

c. Homoscedasticity:
The residuals should have constant variance across all levels of the independent variables. In other words, the spread of residuals should be approximately the same for all values of the independent variables.

d. Normality of Errors:
The residuals should be normally distributed. This is especially important for hypothesis testing and calculating confidence intervals for coefficients.

e. No Multicollinearity (for multiple regression):
In multiple linear regression, the independent variables should not be highly correlated with each other. High correlation between predictors can cause instability in the coefficient estimates.

Handling Heteroscedasticity:
If errors naturally depend on the variable values (heteroscedasticity), we can handle it in several ways:
Transform the Dependent Variable: Apply a transformation (like log or square root) to stabilize the variance.
Use Weighted Least Squares (WLS): Give different weights to data points based on their variance, so the model accounts for varying reliability in predictions.
Use Robust Standard Errors: Adjust the calculation of standard errors to be more reliable in the presence of heteroscedasticity.
Model the Variance: Sometimes, it’s useful to explicitly model how variance changes with the independent variable.

Problem statement:-
Predict the Final Grade Based on Study Time
The objective is to build a simple linear regression model that predicts the final grade (G3) based on the study time (studytime).
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import find_outliers, fix_outliers


# import the csv
df = pd.read_csv("student_mat.csv", delimiter=";")
# Clean the data in case if something is missing.
if df.isna().any().any():
    df = df.interpolate()

fix_outliers(df, 'studytime')
fix_outliers(df, 'G3')

df_relevant = df[['studytime', 'G3']]

X = df_relevant[['studytime']]
y = df_relevant['G3']

# Split the data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model and fit the data
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# calculate the metrics to know the fit of model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"mean_square_error: {mse}")
print(f"mean_square_error: {mae}")
print(f"R2 score: {r2}")

''' Output:-
Fixing outliers: 27
mean_square_error: 21.088970676611243
mean_square_error: 3.657193715640974
R2 score: -0.02847705742452078

results:-
Studytime is not enough to make a linear relationship with G3.
Gonna use multiple linear regression for same.
'''
