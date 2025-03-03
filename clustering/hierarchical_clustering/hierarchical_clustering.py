import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,
    [3, 4]].values  # there is no independent or dependent feature in clustering, removing the array slice of -1
# y = dataset.iloc[:, -1].values
# print("X: ", x)


hierarchical_clustering = sch.dendrogram(sch.linkage(x, method='ward'))  # ward - minimizing the variance in the cluster

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.savefig('Dendrogram.png')

# from the graph, the number of cluster is 5
from sklearn.cluster import AgglomerativeClustering

als = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
als.fit(x)

als_predict = als.fit_predict(x)
print(als_predict)

# plotting the graph
plt.clf()
plt.scatter(x[als_predict == 0, 0], x[als_predict == 0, 1], s=100, color='red', label='cluster 1')
plt.scatter(x[als_predict == 1, 0], x[als_predict == 1, 1], s=100, color='blue', label='cluster 2')
plt.scatter(x[als_predict == 2, 0], x[als_predict == 2, 1], s=100, color='green', label='cluster 3')
plt.scatter(x[als_predict == 3, 0], x[als_predict == 3, 1], s=100, color='orange', label='cluster 4')
plt.scatter(x[als_predict == 4, 0], x[als_predict == 4, 1], s=100, color='violet', label='cluster 5')
plt.title("Hierarchical Clustering")
plt.xlabel('Annual income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('hierarchical.png')

