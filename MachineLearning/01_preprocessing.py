# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
# data.iloc = integer-location based indexing for selection by position.
#  .values gives you the values at the index
data = pd.read_csv('Data.csv')
print data
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
print X, y

# Fill in missing values
# Imputer means replacing missing data with substituted values.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding the Dependent Variable as vector
# OneHotEncoder involves splitting the column which contains categorical data
# to many columns depending on the number of categories present in that column.
# passthrough = all remaining columns that were not specified in transformers will be
# automatically passed through. This subset of columns is concatenated with the output of the transformers.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
# print X

# encodes binary labels (i.e. yes and no) to binary 1s and 0s
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Feature Scaling.
# Standardization (x - mean(x)) / STD(x)
# Use transform on test so that it uses the same scalars as the training using fit_transform
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train[:, 3:] = ss.fit_transform(X_train[:, 3:])
X_test[:, 3:] = ss.transform(X_test[:, 3:])

