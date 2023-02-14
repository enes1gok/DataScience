import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Machine_learning/PolynomialRegression/car-carvelocity.csv",sep=";")

df = df.rename(columns={"araba_fiyat" : "car_price",
                        "araba_max_hiz" : "max_car_velocity"})

y = df.max_car_velocity.values.reshape(-1,1)
x = df.car_price.values.reshape(-1,1)
'''
plt.scatter(x,y)
plt.ylabel("Max Car Velocity")
plt.xlabel("Car Price")
plt.title("Velocity vs Price")
plt.grid(True)
plt.show()
'''
polynomial_regression = PolynomialFeatures(degree=4)
x_polynom = polynomial_regression.fit_transform(x,y)

lr = LinearRegression()
lr.fit(x_polynom, y)
y_head = lr.predict(x_polynom)

'''
plt.scatter(x,y)
plt.plot(x,y_head,color="red",label="polynomial")
plt.legend()
plt.grid(True)
plt.ylabel("Max Car Velocity")
plt.xlabel("Car Price")
plt.title("Relationship Velocity and Price")
plt.show()
'''

#y = b0+b1x+b2x^2+b3x^3...

from sklearn.metrics import r2_score
# R^2 EVALUATION -- It allows r^2 regression models to be evaluated, and the larger r^2 the more accurate the prediction.
print("R^2 of Polynomial Regression model is {}".format(r2_score(y,y_head)))





