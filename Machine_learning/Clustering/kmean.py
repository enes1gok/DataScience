import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#create dataset

#cluster1
x1 = np.random.normal(25,5,1000) #there are 1000 points which the standart deviaiton is 5 and mean is 25
y1 = np.random.normal(25,5,1000)

#cluster2
x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

#cluster3
x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3), axis = 0) #combine = concatenate
y = np.concatenate((y1,y2,y3), axis = 0)

dictionary = {'x': x,'y': y}

data = pd.DataFrame(dictionary)
#print(data.head())
'''
plt.figure()
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("The Dataset for K Mean Cluster Method")
plt.show()'''

#this is how the k-averaging algorithm will see the data
'''plt.figure()
plt.scatter(x1,y1, color="black")
plt.scatter(x2,y2, color="black")
plt.scatter(x3,y3, color="black")
plt.xlabel("x")
plt.ylabel("y")
plt.title("The Dataset for K Mean Cluster Method")
plt.show()'''

#choice of k value          =       We are looking for how many clusters we should create.
#we will use elbow method
from sklearn.cluster import KMeans
wcss = []

for c in range(1,15):
    kmeans =  KMeans(n_clusters=c)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

'''plt.figure()
plt.plot(range(1,15),wcss)
plt.xticks(range(1,15))
plt.xlabel("Number of Clusters (C)")
plt.ylabel("wcss")
plt.show()'''

#For k = 3, we will train, test, and then visualize 3 mean clustering methods.
k_mean = KMeans(n_clusters=3)
clusters = k_mean.fit_predict(data)
data["label"] = clusters

plt.figure()
plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color = "red",label = "Cluster 1")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color = "green",label = "Cluster 2")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color = "blue",label = "Cluster 3")
plt.scatter(k_mean.cluster_centers_[:,0],k_mean.cluster_centers_[:,1],color = "yellow")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("3 mean clustering result")
plt.show()