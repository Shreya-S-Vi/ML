import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

datas = pd.read_csv('Position_Salaries.csv')

print(datas)

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2].values

print("x val:\n",x)
print("y val:\n",y)

lin = LinearRegression()
lin.fit(x,y)

poly = PolynomialFeatures(degree = 3)
x_poly = poly.fit_transform(x)

poly.fit(x_poly,y)
lin = LinearRegression()
lin.fit(x_poly,y)

plt.scatter(x,y,color= 'red')

plt.plot(x, lin.predict(poly.fit_transform(x)), color = 'green')
plt.title('Polynomial Regression')
plt.xlabel('Levels')
plt.ylabel('Salary x 10^6')

plt.show()

