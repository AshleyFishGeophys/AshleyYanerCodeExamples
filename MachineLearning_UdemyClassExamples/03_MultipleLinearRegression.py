import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. All-in
#       Throw in all variables.
#       Prior knowledge.
#       Prepare backward elimination.
# 2. Backward elimination (step-wise regression)
#       Select significance level to stay in the model (0.05)
#       Fit full model with all possible predictors.
#       Consider predictor with highest P-value. If P > Significance Level, go to next step
#       Remove predictor with highest P-value
#       Fit model without variable
# 3. Forward selection (step-wise regression)
#       Select significance level to enter the model (0.05)
#       Fit all simple regression models. Select the lowest P-value
#       Keep variable and fit all possible models with one extra predictor added
#       to the ones you already have.
#       Consider the predictor with the lowest p-value. If P < Significance Level, go to previous step.
#       Keep previous model.
# 4. Bidirectional Elimination (step-wise regression. Most common.)
#       Select a significance level to enter and stay in the model.
#       Preform next step of Forward selection.
#       Perform all steps of Backward elimination.
#       No new variables can enter and no old variables can exit.
# 5. All possible models
#       Select a criterion of goodness of fit
#       Construct all possible regression models
#       Select the one with the best criterion
#       Resource consuming
# 6. Score comparison

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python'
data = pd.read_csv(path + r'\50_Startups.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

pd.set_option('display.max_rows',100)
print(data)

# Encode categorical data. Dummy variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
X = ct.fit_transform(X)
# print(X)

# No need to apply feature scaling

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,train_size=0.2, random_state=0)

# Linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

# Predict test results in vectors
# X_pred = lr.predict(y_test)
y_pred = lr.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)), 1))

plt.scatter(y_pred, y_test)
plt.show()

# Evaluate the Model performance
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))