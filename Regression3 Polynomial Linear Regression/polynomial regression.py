# Polynomial regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
'''
not needed here as our dataset is too small

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

'''

#fitting linear regression in dataset (building two models just to compare. no need to create this)
from sklearn.linear_model import LinearRegression
linearRegressor=LinearRegression()
linearRegressor.fit(x, y)

#fitting polynomial regression in dataset
#polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
polynomialRegressor=PolynomialFeatures(degree=2)
xPolynomial=polynomialRegressor.fit_transform(x)
linearRegressor2=LinearRegression()  #making this linear regression object to fit our polynomial x in to the linear regressor model
linearRegressor2.fit(xPolynomial, y)

#Visulaizing linear regression
plt.scatter(x, y,color='red')
plt.plot(x,linearRegressor.predict(x),color='blue')
plt.title("Truth or Bluff (Linear regression model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Visulaizing polynomial linear regression
plt.scatter(x, y,color='red')
plt.plot(x,linearRegressor2.predict(polynomialRegressor.fit_transform(x)),color='blue')
plt.title("Truth or Bluff (Linear regression model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

















