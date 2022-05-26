#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset= pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values='Nan',strategy='mean',axis=0)
imputer.fi