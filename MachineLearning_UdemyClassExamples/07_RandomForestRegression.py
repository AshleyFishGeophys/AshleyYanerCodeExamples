import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensemble learning - put together multiple algorithms
# 1. Pick at random K data points from Training set
# 2. Build the Decision Tree associated with those K data points
# 3. Choose the number Ntree of trees you want to build and repeat steps 1 and 2.
# 4. For a new data point, make each one of your Ntree trees predict the value
# of Y for the new data point in questions. Assign the new data point
# the average aceoss all of the predicted Y values.

path =r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 9 - Random Forest Regression\Python'

df = pd.read_csv(path+r'\Position_Salaries.csv')
# print df.head()

X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

# Train the random forest regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y)

# predict value
print regressor.predict([[6.5]])

# Display results
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()