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
import statsmodels.api as sm

'''
AUTOMATIC BACKWAR ELIMINATION MODEL
'''
def backwardElimination(x, sl):
    numberVariables = len(x[0])
    for i in range(0, numberVariables):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numberVariables - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_optimal = x[:, [0, 1, 2, 3, 4, 5]]
x_optimal = np.array(x_optimal, dtype=float)
x_Modeled = backwardElimination(x_optimal, SL)




'''
MANUAL BACKWARD ELIMINATION

x=np.append(arr=np.ones((50, 1)).astype(int), values=x,axis=1)
x_optimal=x[:,[0,1,2,3,4,5]]
x_optimal = np.array(x_optimal, dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=x_optimal).fit()

regressor_OLS.summary()

#removing the highest p value
x_optimal=x[:,[0,1,3,4,5]]
x_optimal = np.array(x_optimal, dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=x_optimal).fit()

regressor_OLS.summary()

#removing the highest p value
x_optimal=x[:,[0,3,4,5]]
x_optimal = np.array(x_optimal, dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=x_optimal).fit()

regressor_OLS.summary()

#removing the highest p value
x_optimal=x[:,[0,3,5]]
x_optimal = np.array(x_optimal, dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=x_optimal).fit()

regressor_OLS.summary()

#removing the highest p value
x_optimal=x[:,[0,3]]
x_optimal = np.array(x_optimal, dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=x_optimal).fit()

regressor_OLS.summary()
'''
















