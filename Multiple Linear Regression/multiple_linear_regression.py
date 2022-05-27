# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoderCountry=LabelEncoder()
x[:,3]=labelEncoderCountry.fit_transform(x[:,3]) #encoding the data of column1=state
#creating the dummy variable to change the order
columnTransformer=ColumnTransformer([("State",OneHotEncoder(),[3])],remainder='passthrough')#declaring the column number for which we need to use the encoder 
x=columnTransformer.fit_transform(x) #fitting and transforming the datasetx
 
  




















