# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoderState=LabelEncoder()
x[:,3]=labelEncoderState.fit_transform(x[:,3]) #encoding the data of column1=state
#creating the dummy variable to change the order
columnTransformer=ColumnTransformer([("State",OneHotEncoder(),[3])],remainder='passthrough')#declaring the column number for which we need to use the encoder 
x=columnTransformer.fit_transform(x) #fitting and transforming the datasetx

#avoiding the dummy variable trap
x=x[:,1:] #avoiding first column, but python linear regression lib can already handle this dont need to inlcude this

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

#prediction of the test result
y_prediction=regressor.predict(x_test)


#Building the model with backward elimination
import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm2

x=np.append(arr=np.ones((50, 1)).astype(int), values=x,axis=1)
x_optimal=x[:,[0,1,2,3,4,5]]
x_optimal = np.array(x_optimal, dtype=float)
regressor_OLS=sm2.OLS(endog=y, exog=x_optimal).fit()


  




















