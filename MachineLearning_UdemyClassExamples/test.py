import pandas as pd
import matplotlib.pyplot as plt

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Python'
data = pd.read_csv(path + '\Position_Salaries.csv')

print data.head(10)

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

# reshape y

# Feature Scaling

# train SVN

# predict SVM

# Plot SVM