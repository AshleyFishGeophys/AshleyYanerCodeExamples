import pandas as pd
import matplotlib.pyplot as plt

# Choose number of K clusters
# Select at random k points, the centroids
# Assign each data point to the closest centroid - that forms k clusters
# Compute and place the new centroid of each cluster
# Reassign each data point to the new closest centroid. Center of mass.
# If any reassignment took place, go back a step. Otherwise converged.
# Choose the right number of clusters - WCSS

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 4 - Clustering\Section 24 - K-Means Clustering\Python'

df = pd.read_csv(path + r'\Mall_Customers.csv')
# print df.head()

# No dependent variable, here.
X = df.iloc[:,3:].values
# print X

# Use elbow method to determine number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i, init = 'k-means++', random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

# plt.plot(range(1,11), wcss)
# plt.title('elbow method')
# plt.xlabel('number of clusters')
# plt.ylabel('WCSS')
# plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
# create dependent variable
y_pred = kmeans.fit_predict(X)
print(y_pred)


plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label= 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label= 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label= 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'cyan', label= 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'purple', label= 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
