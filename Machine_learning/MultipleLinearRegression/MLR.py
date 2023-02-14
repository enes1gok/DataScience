import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Machine_learning/MultipleLinearRegression/R&D-Expense-Marketing.csv")
#print(df.head())
df = df.rename(columns={"ArgeHarcamasi"      : "R&D_Expenditure",
                        "YonetimGiderleri"   : "Management_Expenses",
                        "PazarlamaHarcaması" : "Marketing_Expenditure",
                        "Sehir"              : "City",
                        "Kar"                : "Profit"})

x = df.iloc[:,[0,1,2]].values   #INDEPENDENT VARIABLES
#print(x)
y = df.Profit.values.reshape(-1,1)
#print(y)

multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x,y)

test_datas = np.array([[20.500,20.500,20.500]])
test_result = multiple_linear_reg.predict(test_datas)
#print("If all expenses are {}, our profit is {}".format(test_datas[0],test_result))

from sklearn.metrics import r2_score
# R^2 EVALUATION -- It allows r^2 regression models to be evaluated, and the larger r^2 the more accurate the prediction.
print("R^2 of Multiple Linear Regression model is {}".format(r2_score(y,test_result)))


