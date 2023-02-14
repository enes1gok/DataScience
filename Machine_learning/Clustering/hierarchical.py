import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#create dataset

#cluster1
x1 = np.random.normal(25,5,20) #there are 20 points which the standart deviaiton is 5 and mean is 25
y1 = np.random.normal(25,5,20)

#cluster2
x2 = np.random.normal(55,5,20)
y2 = np.random.normal(60,5,20)

#cluster3
x3 = np.random.normal(55,5,20)
y3 = np.random.normal(15,5,20)

x = np.concatenate((x1,x2,x3), axis = 0) #combine = concatenate
y = np.concatenate((y1,y2,y3), axis = 0)

dictionary = {'x': x,'y': y}

data = pd.DataFrame(dictionary)

'''plt.figure()
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("The Dataset for Hierarchical Cluster Method")
plt.show()'''

#dendogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data, method = "ward")
#dendrogram(merg, leaf_rotation=90)
'''plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()'''

#We will train the Hierarchical Clustering method to create 3 clusters, test it and then visualize it.

from sklearn.cluster import AgglomerativeClustering

hierarchy_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean')
cluster = hierarchy_cluster.fit_predict(data)
data['label'] = cluster

plt.figure()
plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color = "red",label = "Cluster 1")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color = "green",label = "Cluster 2")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color = "blue",label = "Cluster 3")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("3 Mean Clustering Result")
plt.show()