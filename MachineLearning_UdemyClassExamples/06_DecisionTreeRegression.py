import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Information entropy. splits - terminal leaves.

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 8 - Decision Tree Regression\Python'

df = pd.read_csv(path+r'\Position_Salaries.csv')
# print df.head(25)

X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

# print X, y

# Train the decision tree regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

# Predict new result
print regressor.predict([[6.5]])

# Visualization of the decision tree regression results.
# It's 2D so we can visualize easily.
# It gets more difficult with higher dimensions.
# Decision Trees are more adaptable to higher dimensions. 2D isn't great.
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()