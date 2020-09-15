import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 6 - Polynomial Regression\Python'

data = pd.read_csv(path + r'/Position_Salaries.csv')
# print data
X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

# print data.head(10)
# print X, y

# Linear regression model for comparison to polynomial
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

# Polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=4)
X_poly = pr.fit_transform(X)
lr_poly = LinearRegression()
lr_poly.fit(X_poly,y)

# # Visualize Linear Regression results
# plt.scatter(X,y, color = 'red')
# plt.plot(X,lr.predict(X), color = 'blue')
# plt.title('Linear regression model')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Visualize Polynomial Regression results. Higher res
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lr_poly.predict(pr.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial regression model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predict new result
# print lr.predict([[6.5]])
# print lr_poly.predict(pr.fit_transform([[6.5]]))

# Evaluate the Model performance
