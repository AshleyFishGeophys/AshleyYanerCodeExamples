import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
classical(classification) and deep learning
if/else chatbot
audio frequency comonents analysys - speech recognition
Bag of words model - classification. Much like google email reply suggestions
Convolutional neural networks for text recognition - classification (image recognition)
seq2seq - many applications
'''

# Bag of words model

# Import restaurant review data
path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Python'
df = pd.read_csv(path+'\Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # quoting = 3 -> Ignore quotes

print(df.head())

# Clean the text
import re
import nltk #Used for stop words - and. uh, if, etc
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #apply stemming on reviews. i.e loved = love. root of word
corpus = [] # A large and structured set of machine-readable texts that have been produced in a natural communicative setting

for i in range(0,1000): #1000 reviews
    review = re.sub('[^a-zA-z]', ' ', df['Review'][i])  # remove all punctuations. Substitute anything that's not a letter with a space
    review = review.lower() # Makes everything lowercase
    review = review.split() # Splits into different words
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in all_stopwords] #Simplify words to root of word
    review = ' '.join(review) #Combine cleaned words
    corpus.append(review)

# print(corpus)

# Create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #All words into columns
y = df.iloc[:,-1].values

print(len(X[0])) #Figure out how many words are used for the max_features in cv method

# Train test split
from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(X,y, random_state = 0, test_size=0.2)

# Train Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predict test results
y_pred = classifier.predict(X_test)

# Confustion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))




