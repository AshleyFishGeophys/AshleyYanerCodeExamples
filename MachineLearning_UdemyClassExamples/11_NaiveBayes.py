import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Naive Bayes Theory: P(A|B) = P(B|A) * P(A) / P(B)

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 3 - Classification\Section 16 - Support Vector Machine (SVM)\Python'
df = pd.read_csv(path + r'\Social_Network_Ads.csv')

# print df.head(25)

X = df.iloc[:,2:-1].values
y = df.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)

# Confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
print confusion_matrix(y_test, y_pred)
print accuracy_score(y_test, y_pred)


K
