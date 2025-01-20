''' Application of binary classification on famous titanic survival
dataset.
TODO:  Focus on softmax after this... dont forget.
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from utils.utils import fix_outliers


df = pd.read_csv("datasets/titanic.csv")

# Removing the passengerId as it does not make sense to include it
df = df.drop(columns=['PassengerId'])

# Missing Data fixing if any
if df.isna().any().any():
    for column in df.select_dtypes(include=[np.number]):
        df[column].fillna(df[column].median(), inplace=True)
    for column in df.select_dtypes(include=["object", "category"]):
        df[column].fillna(df[column].mode()[0], inplace=True)

y = df["Survived"]
X = df.drop(columns=["Survived"])


encoder = LabelEncoder()
X['Sex'] = encoder.fit_transform(X['Sex'])

# Fix the outliers
# for column in X.select_dtypes(include=[np.number]):
#     X[column] = fix_outliers(X, column)

# Standardize the data to neutralize the effect from larger ranges
scaler = StandardScaler()
X_stan = scaler.fit_transform(X)

X = pd.DataFrame(X_stan)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

model = LogisticRegression(max_iter=200, solver="liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion: {confusion}")
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")

'''
Accuracy: 0.8202247191011236
Confusion: 
[[98 12]
[20 48]]
F1 Score: 0.75
Recall: 0.7058823529411765
Precision: 0.8
'''