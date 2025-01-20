''' 
Supervised Learning:

Definition:
Supervised learning is a type of machine learning where the model learns from labeled data. The data consists of input-output pairs, 
 where the output (label) is known.
The goal is to learn a mapping function from inputs to outputs so that the model can predict the label for new, unseen data.

The process involves two main phases:

    Training: The model is trained on a dataset with labeled examples.
    Prediction: The trained model is used to predict the output for new, unseen input data.

Task for Practice:

For practice, I suggest the following steps:

    Get a simple dataset with labeled data, such as the Iris dataset or Boston Housing dataset.
    Load the dataset and inspect it. Look at a few rows of data and identify the features (inputs) and target labels (outputs).
    Try splitting the dataset into a training set and a test set (e.g., 80% for training and 20% for testing).
    Write code to implement a simple model (e.g., Linear Regression) to predict the target variable using the training set.
    Evaluate the model's performance on the test set.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the iris data with labels
data = load_iris()

# Get the data in format 
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200) # So that the model converges properly.
model.fit(X_train, y_train)

# Run the model on testing set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy}")


species_names = data.target_names
y_pred_names = species_names[y_pred]
y_test_names = species_names[y_test]

# Plotting the results
plt.scatter(X_test['sepal length (cm)'], y_test_names, color='blue', label='Actual')
plt.scatter(X_test['sepal length (cm)'], y_pred_names, color='red', label='Predicted')
plt.title("Logistic Regression: Actual vs Predicted")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Species")
plt.legend()
plt.show()
