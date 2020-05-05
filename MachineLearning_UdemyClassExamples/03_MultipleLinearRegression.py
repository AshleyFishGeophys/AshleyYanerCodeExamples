import pandas as pd
import matplotlib.pyplot as plt

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 5 - Multiple Linear Regression\Python'
data = pd.read_csv(path + r'\50_Startups.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

print data