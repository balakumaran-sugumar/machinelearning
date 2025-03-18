import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# import the data  (step 1)
# remove the quotes to avoid model fitting
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
# print(dataset.head(10))

# cleaning the text (step 2)
import re
import nltk
nltk.download('stopwords') # to avoid including non-relevant words in review (the, in, a, an)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # simplifies words e.g. loved => love, to reduce final dimension

# cleaned data
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()

    # applying steaming
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# print(corpus)

# creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_features=1500)
X = count_vect.fit_transform(corpus).toarray()
# print(len(X[0]))
print(X[0])
y = dataset.iloc[:, -1].values
print(y)

# training the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classification = LogisticRegression(random_state=0)
classification.fit(X_train,y_train)

predict = classification.predict(X_test)

print("Actual vs Predicted: ")
print(np.concatenate((predict.reshape(len(predict), 1), y_test.reshape(len(y_test), 1)), 1))

# calculating the accuracy and the conf matrix
accuracy = accuracy_score(y_test, predict)
print("Accuracy: ", accuracy)

confusion_matrix = confusion_matrix(y_test, predict)
print("Confusion Matrix: ", confusion_matrix)