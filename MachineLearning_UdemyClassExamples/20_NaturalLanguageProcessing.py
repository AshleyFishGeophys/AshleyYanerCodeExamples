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
df = pd.read_csv(path+'\Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# print df.head()

# Clean the text
import re
import nltk #Used for stop words - and. uh, if, etc
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #apply stemming on reviews. i.e loved = love

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i]) #remove all punctuation. Replace everything which is not a letter with a space.
    review = review.lower() #Transform all capitals to lower case
    review = review.split() #splits reviews into different words
    ps = PorterStemmer() #stemming. simplify each word with rout of word. i.e. loved = love
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not') #Exclude not in stopwords
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] #get rid of stop words
    review = ' '.join(review) #Add spaces between each word in word list
    corpus.append(review)

# print corpus


# Create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values
# print len(X[0])


# Split into training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.30, random_state=0)

# Split

