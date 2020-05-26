import pandas as pd
import matplotlib.pyplot as plt

# Assumptions:
# 1. Linearity -
#       is the property of a mathematical relationship or function which means that it can be graphically
#       represented as a straight line. Examples are the relationship of voltage and current across a resistor
#       (Ohm's law), or the mass and weight of an object. Avoid the dummy variable trap where one variable
#        predicts another variable. Include all but one dummy variable.
# 2. Homoscedaticity -
#       describes a situation in which the error term (that is, the "noise" or random
#       disturbance in the relationship between the independent variables and the dependent variable)
#       is the same across all values of the independent variables. Same variance.
# 3. Independent of errors -
#       If your points are following a clear pattern, it might indicate that the errors are
#       influencing each other. The errors are the deviations of an observed value from the true function value.
# 4. Lack of multicollinearity -
#       inflates the variances of the parameter estimates. This may lead to lack of
#       statistical significance of individual independent variables even though the overall model may be significant.
#       Such problems will result in incorrect conclusions about relationships between independent and dependent variables.


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
y_test_predict = regressor.predict(X_test)
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
