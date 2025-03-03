import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3, 4]].values  # there is no independent or dependent feature in clustering, removing the array slice of -1
# y = dataset.iloc[:, -1].values
# print("X: ", x)

# for purpose of this exercise, keeping annual income and spending score
from sklearn.cluster import KMeans

# create upto 10 clusters
wcss = []

for i in range(1, 31):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # to avoid random initialization trap
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 31), wcss, label='K Means++')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('kmeans-elbow.png')

k_means = KMeans(n_clusters=5, init='k-means++',
                 random_state=42)  # will contain 5 groups of customers - have dependent and independent variable
k_means.fit(x)
y_kmeans = k_means.predict(x)  # helps create the dependent variable here

plt.clf()
# print(y_kmeans)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, color='red', label='cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, color='blue', label='cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, color='green', label='cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, color='orange', label='cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, color='violet', label='cluster 5')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=300, color='yellow', label='centroids')
plt.title("Centroids of K-Means")
plt.xlabel('Annual income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('K_Mean_clusters.png')