import pandas as pd

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 3 - Classification\Section 14 - Logistic Regression'

df = pd.read_csv(path+r'\breast-cancer-wisconsin_full.csv')

# print df.head(25)

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

print X
# print X,y

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predict test results
y_pred = classifier.predict(X_test)

# Confusion matrix
from sklearn.metrics import classification
cm = classifier(X_test, y_pred)
print cm

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))