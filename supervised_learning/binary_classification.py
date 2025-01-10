''' Logistic Regression for binary classification.
The model predicts the probability that a given input point belongs to one of the two categories, using the sigmoid function to output values between 0 and 1:
P(y=1∣X)=1- (1+e^(−z))

Where:

    z is the linear combination of the input features.
    e is the base of the natural logarithm
    P(y=1∣X)P(y=1∣X) is the probability of belonging to class 1

z = w1​⋅X1​+w2​⋅X2​+⋯+wn​⋅Xn​+b
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

from utils import fix_outliers

df = pd.read_csv("breast_cancer.csv")

# Fix the missing data inputs
if df.isna().any().any():
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].median(), inplace=True)
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

# Seperate the features and target values
y = df['diagnosis']
X = df.drop(columns=['diagnosis'])

# Convert Non numerical values to encodings.
for column in X.select_dtypes(include=["object", "category"]):
    X[column] = LabelEncoder().fit_transform(X[column])

# Fix the outliers
for column in X.select_dtypes(include=[np.number]):
    fix_outliers(df, column)

# Standardize the Features so that no particular features with large
# range dominate the learning process
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X = pd.DataFrame(X_standardized)

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Get the accuracy of the model predictions
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")