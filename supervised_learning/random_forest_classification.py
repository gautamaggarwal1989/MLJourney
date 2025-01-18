'''
1. What is Random Forest? Random Forest is an ensemble learning method used for classification and regression tasks. It operates by constructing multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.

2. Key Concepts:

    Ensemble Methods: Random Forest is an ensemble method that combines the predictions of multiple models to improve accuracy. In this case, it uses decision trees.
    Bagging (Bootstrap Aggregating): Each tree in the forest is trained on a random subset of the data (with replacement). This process helps reduce variance and avoid overfitting.
    Feature Importance: Random Forest can estimate the importance of each feature in making predictions by analyzing how much the prediction error increases when a feature's values are permuted.

3. How it Works:

    Training: A random subset of the training data and features is used to build each tree.
    Prediction: For classification, the majority vote of the trees is taken. For regression, the average of the predictions is used.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.utils import fix_outliers

df = pd.read_csv("datasets/titanic.csv")

# Remove the PassengerId column as it has no role in predicitions
df = df.drop(columns=["PassengerId"])

if df.isna().any().any():
    for column in df.select_dtypes(include=np.number).columns:
        df[column].fillna(df[column].median(), inplace=True)
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

y = df['Survived']
X = df.drop(columns=["Survived"])

# Fix the outliers
for column in X.select_dtypes(include=np.number):
    X[column] = fix_outliers(X, column)

# Encoding the categorical data to numerical representation
for column in X.select_dtypes(include=["category", "object"]):
    X[column] = LabelEncoder().fit_transform(X[column])

# Standardize the output
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Convert X_s back to pd Dataframe
X = pd.DataFrame(X_s)

# Split the training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# collect the predictions
y_pred = model.predict(X_test)

# Get metrics and print them
accuracy = accuracy_score(y_test, y_pred)
c_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report: {c_report}")

# Get the importance score of each feature in prediction
features = X.columns
importance = model.feature_importances_
for i, v in enumerate(importance):
    print(f"Feature Index: {features[i]}, Score: {v}")

'''
Output:-
Fixing outliers: 46
Fixing outliers: 213
Fixing outliers: 114
Accuracy: 0.797752808988764
Classification Report:               precision    recall  f1-score   support

           0       0.85      0.82      0.83       109
           1       0.73      0.77      0.75        69

    accuracy                           0.80       178
   macro avg       0.79      0.79      0.79       178
weighted avg       0.80      0.80      0.80       178

Feature Index: 0, Score: 0.0955104686839914
Feature Index: 1, Score: 0.26151131449948767
Feature Index: 2, Score: 0.28664005409701204
Feature Index: 3, Score: 0.045196531922925845
Feature Index: 4, Score: 0.0
Feature Index: 5, Score: 0.27224252530662985
Feature Index: 6, Score: 0.038899105489953345
'''