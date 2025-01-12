'''
Multinomial regression is a category or extension of Logistic regression where
target variable can assume more than two classifaction values.
A probability is assigned to each class with respect to an observation and class
with highest probability is assigned to it. Sum of all the probability distribution
comes to 1.

The performance can often be improved by tuning hyperparameters, handling class imbalances,
feature engineering, or using more complex models.
'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.utils import fix_outliers

df = pd.read_csv("datasets/winequality-red.csv")

# Clean the data by replacing the missing data with 
if df.isna().any().any():
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].median(), inplace=True)
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

# All values are numerical so there is no need for encoding
# Find and fix the outliers using IQR

y = df['quality']
X = df.drop(columns=['quality'])

for column in X:
    X[column] = fix_outliers(X, column)

# Standardizing the data so that features with wide range do not create a bias
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X = pd.DataFrame(X_standardized)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(
    solver='liblinear', max_iter=200
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate the confusion matrix and accuracy score
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print(f"Confusion Matrix: {confusion}")

'''Output
Accuracy Score: 0.58125
Confusion Matrix: 
[[ 0  0  1  0  0]
[ 0  1  7  2  0]
[ 0  0 97 33  0]
[ 0  0 46 73 13]
[ 0  0  2 30 15]]
'''
