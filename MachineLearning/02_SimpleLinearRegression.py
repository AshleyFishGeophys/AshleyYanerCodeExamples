import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 4 - Simple Linear Regression\Python'

# Import data
data = pd.read_csv(path + '\Salary_Data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# plt.scatter(X, y)
# plt.show()

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33, random_state=0)

# Train
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predict
y_test_redict = regressor.predict(X_test)
y_train_predict = regressor.predict(X_train)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_train_predict, color = 'blue')
plt.title('Salary vs Experience (Training set)') # Regression line
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, y_train_predict, color = 'blue') # Regression line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
