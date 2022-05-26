#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset= pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values #taking all the dataset till the last column
y=dataset.iloc[:,3].values #taking the data of the last column. Used :3 as in python it iterates till the previous value of x where x is the value inserted after:. like :x

#Taking care of missing data
from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.NAN,strategy='mean',fill_value=None,verbose=0,copy=True) #used imputer to replace the missing values with the mean of the column
imputer=imputer.fit(x[:,1:3]) #taking the column number one and two
x[:,1:3]=imputer.transform(x[:,1:3]) #transforming the column number

#Encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoderCountry=LabelEncoder()
x[:,0]=labelEncoderCountry.fit_transform(x[:,0]) #encoding the data of column1=country
#creating the dummy variable to change the order
columnTransformer=ColumnTransformer([("Country",OneHotEncoder(),[0])],remainder='passthrough')#declaring the column number for which we need to use the encoder 
x=columnTransformer.fit_transform(x) #fitting and transforming the datasetx

#for the purchase column
labelEncoderPurchase=LabelEncoder()
y=labelEncoderPurchase.fit_transform(y)








