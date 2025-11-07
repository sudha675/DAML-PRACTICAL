# Program 16: Hierarchical Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
iris = pd.read_csv("IRIS.csv")

# Select features
x = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(x)

# Visualize the clusters
plt.figure(figsize=(7, 5))
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=50, c='green', label='Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=50, c='blue', label='Cluster 3')

plt.title('Hierarchical Clustering on IRIS Dataset')
plt.xlabel('Feature 1 (e.g. Sepal Length)')
plt.ylabel('Feature 2 (e.g. Sepal Width)')
plt.legend()
plt.show()

# Dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linkage(x, method='ward'))
plt.title('Dendrogram for IRIS Dataset')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distances')
plt.show()
