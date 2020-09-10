import pandas as pd
import matplotlib.pyplot as plt

# Make each data point a single cluster - N clusters
# Take the two closest data points and make them one cluster - N-1 clusters
# Take the two closest clusters and make them one cluster - N-2 clusters
# Repeat previous stem until there is only one cluster

# Distance between two clusters:
# 1. Closest points
# 2. Furthest points
# 3. Average distances
# 4. Distance between centroids


path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 4 - Clustering\Section 25 - Hierarchical Clustering\Python'

df = pd.read_csv(path + r'\Mall_Customers.csv')
print df.head()

# No dependent variable, here.
X = df.iloc[:,3:].values

# Dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Fit model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'ward')
y_pred = hc.fit_predict(X)
print y_pred

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label= 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label= 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label= 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'cyan', label= 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'purple', label= 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
