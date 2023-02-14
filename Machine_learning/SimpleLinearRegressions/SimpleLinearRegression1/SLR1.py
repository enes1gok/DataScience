import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Machine_learning/SimpleLinearRegressions/SimpleLinearRegression1/Experience-Wage.csv", sep = ",")

df= df.rename(columns={"Kidem":"Experience",
                        "Maas":"Wage"})

linear_reg = LinearRegression()
x = df.Experience.values.reshape(-1,1)
y = df.Wage.values.reshape(-1,1)
linear_reg.fit(x,y)

b0 = linear_reg.intercept_
b1 = linear_reg.coef_

experience = 10
result = linear_reg.predict(np.array([experience]).reshape(1,-1))
#print("{} years experience = {} wage amount".format(experience,result[0]))


array = np.array([11:30]).reshape(-1,1)
y_head = linear_reg.predict(array)
'''
plt.figure()
plt.scatter(x,y)
plt.plot(array,y_head,color="b")
plt.xlabel("Experience")
plt.xlabel("Wage")
plt.title("Relationship between Experience and Wage")
plt.grid(True)
plt.show()
'''

from sklearn.metrics import r2_score
# R^2 EVALUATION -- It allows r^2 regression models to be evaluated, and the larger r^2 the more accurate the prediction.
print("R^2 of Simple Linear Regression model is {}".format(r2_score(y,y_head)))