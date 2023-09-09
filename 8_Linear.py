import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,
                                                 random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,y_test,color="Red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title("Salary VS Experience(Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
