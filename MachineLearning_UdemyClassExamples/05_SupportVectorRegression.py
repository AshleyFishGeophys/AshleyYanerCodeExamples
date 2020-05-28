import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Epsilon insensitive tube
# Minimize slack veriables xi error
# Support vectors are outside epsilon tube

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python'
data = pd.read_csv(path + '\Position_Salaries.csv')
print data.head(10)

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values
# print X,y

# reshape y so that feature scaling works properly
y = y.reshape(len(y),1)
# print y

# Feature Scaling
# Need to apply since y is has much smaller values than X
# Need to apply to X to be in the same range as y
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
# print X,y

# Train SVR on whole dataset, since data is too small for split (10 rows)
# Gaussian RBF(radial base function) kernal. Lots of kernels available, though
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
# regressor = SVR(kernel = 'poly')
# regressor = SVR(kernel = 'linear')
regressor.fit(X,y)

# Predict new result
# Need to reverse feature scaling to see the correct prediction relative to original data
print sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# # Plot to SVR
# # SVR didn't catch the outlier at the end
# plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color = 'red')
# plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
# plt.show()

# Higher resolution graph
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.show()
