import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Simple Linear regression

df = pd.read_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Machine_learning/SimpleLinearRegressions/SimpleLinearRegression2/SAT-GPA.csv", sep = ",")
#print(df.head())
df= df.rename(columns={"SAT":"sat",
                       "GPA":"gpa"})

#Firstly, you should visualize the data to see what it is.
'''
plt.scatter(df.sat,df.gpa)
plt.xlabel("Scholastic Assessment Test")
plt.ylabel("Grade Point Average")
plt.title("Relationship between SAT and GPA")
plt.grid(True)
plt.show()
'''
linear_reg = LinearRegression()

# In scikit learn, we need to convert the columns in data to numpy array.
x = df.sat.values.reshape(-1,1)
y = df.gpa.values.reshape(-1,1)

linear_reg.fit(x,y)

#Now, I find the point where it cuts the y-axis.
b0 = linear_reg.intercept_
#print("b0: ",b0)

#Now, I find the slope.
b1 = linear_reg.coef_
#print("b1: ",b1)

#Linear Regression model(Prediction)
sat = 1966
result = linear_reg.predict(np.array([sat]).reshape(1,-1))
#print("If SAT score: {}, GPA should be: {}".format(sat,result[0]))

#Visualization
array = np.array([1650,1675,1700,1725,1750,1775,1800,1825,1850,1875,1900,1925,1950,1975,2000,2025,2050]).reshape(-1,1)

plt.figure()
plt.scatter(x,y)
y_head = linear_reg.predict(array) #y_head = GPA
plt.plot(array,y_head, color = "y")
plt.xlabel("Scholastic Assessment Test")
plt.ylabel("Grade Point Average")
plt.title("Relationship between SAT and GPA")
plt.grid(True)
plt.show()






