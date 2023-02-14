import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Machine_learning/Classification/column_2C_weka.csv")
#print(data.head())
'''
#Visualize what classes you have.
sns.countplot(data = data, x="class")
plt.show()
'''
#Let me fill the class column with ones and zeros.
data["class"] = [1 if each == 'Abnormal' else 0 for each in data['class']]
#print(data.head())
#print(data.info()) #So far all variables were numeric!!!

y = data['class'].values # We put the classes inside the variable y.
x_data = data.drop(['class'],axis=1)
'''
sns.pairplot(x_data)
plt.show()
'''

#Normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#training test segmentation
from sklearn.model_selection import train_test_split

# 15% test, 85% train
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=1) # random state provides repeatability.

#we get transpose.
#x_train = x_train.T
#x_test = x_test.T
#y_train = y_train.T
#y_test = y_test.T

#print("x_train: ",x_train.shape)
#print("x_test: ",x_test.shape)
#print("y_train: ",y_train.shape)
#print("y_test: ",y_test.shape)

#Now we will train a logistic regression model, and then actualize its test.
from sklearn.linear_model import LogisticRegression

#train
lr = LogisticRegression()
lr.fit(x_train, y_train)

#test
test_accuracy = lr.score(x_test, y_test)
#print("test accuracy: {}".format(test_accuracy))

#K nearest neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
neighbor_amount = 4
knn = KNeighborsClassifier(n_neighbors=neighbor_amount)
knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
#print(" {} Test accuracy of Nearest Neighbor Model: {}".format(neighbor_amount,knn.score(x_test,y_test)))

#Finding best K
score_list = []
for each in range(1,50):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,50),score_list)
plt.xlabel("k values")
plt.ylabel("Accuracy")
plt.title("Finding best K value")
plt.show()