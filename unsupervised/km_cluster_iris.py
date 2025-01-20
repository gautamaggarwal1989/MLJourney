'''
K-Means is one of the most popular and straightforward clustering algorithms. It partitions the dataset into K distinct clusters by minimizing the variance within each cluster.
Steps Involved:

Choose the number of clusters K.
Initialize K centroids randomly.
Assign each data point to the nearest centroid, forming K clusters.
Compute new centroids as the mean of the points in each cluster.
Repeat steps 3 and 4 until centroids no longer change significantly or a maximum number of iterations is reached.

You will use the Iris dataset, which is a common dataset in clustering problems. It contains measurements for 150 iris flowers from three different species (setosa, versicolor, virginica), with four features: sepal length, sepal width, petal length, and petal width.
'''

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from utils.outliers import fix_outliers

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Fix the outliers
for column in df.select_dtypes(include=[np.number]):
    df[column] = fix_outliers(df, column)

X = df.values
# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize the model
# inertia = []
# # Calculate the inertia values for range 1 to 10: Elbow method
# for K in range(1, 11):
#     model = KMeans(n_clusters=K, random_state=42)
#     model.fit(X)

#     inertia.append(model.inertia_)

# plt.scatter(range(1, 11), inertia, marker='o')
# plt.title("Elbow Search")
# plt.xlabel("Number of clusters")
# plt.ylabel("Inertia")
# plt.show()

# # Make the prediction for labels on data
K = 3 # Change this after observing the elbow point
model = KMeans(n_clusters=K, random_state=42)
model.fit(X)
labels = model.predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title(f"K-means Cluster for iris dataset with K: {K}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
