# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #x represents the columns which are independent variables
y = dataset.iloc[:, -1].values #y is the columns that are dependent variables

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set result
#creating a variable that will store all the predictions
y_prediction=regressor.predict(X_test)

#visualizing training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experience(Training set)',)
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#visualizing test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('salary vs experience(Training set)',)
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
