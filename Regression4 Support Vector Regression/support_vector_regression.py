# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:42:42 2022

@author: farha
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values 

# Splitting the dataset into the Training set and Test set
'''
not needed here as our dataset is too small

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

'''
#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaleX=StandardScaler()
scaleY=StandardScaler()
x=scaleX.fit_transform(x)
y=scaleY.fit_transform(y)


#fitting the SVR in dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)


#predicting the result with regression
yPrediction=scaleY.inverse_transform(np.array(regressor.predict(scaleX.transform([[6.5]]))))


#Visulaizing the Support vector regression results
plt.scatter(x, y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title("Truth or Bluff (SVR model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()