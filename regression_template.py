# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:11:53 2022

@author: g5
"""

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

#fitting the regression model in dataset

'''
code the model here

'''

#predicting the result with regression
yPrediction=regressor.predict([[6.5]])


#Visulaizing the regression model
plt.scatter(x, y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title("Truth or Bluff (Linear regression model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()



#Visulaizing the regression model (For higher resolution and smoother graph)
xGrid=np.arange(min(x),max(x),0.1)
xGrid=xGrid.reshape((len(xGrid),1))

plt.scatter(x, y,color='red')
plt.plot(xGrid,regressor.predict(xGrid),color='blue')
plt.title("Truth or Bluff (Linear regression model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

    














