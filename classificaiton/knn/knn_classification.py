import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# scaling the feature set
X_train_scaler = StandardScaler()
x_scaled = X_train_scaler.fit_transform(X_train)

X_test_scaler = StandardScaler()
x_test_scaled = X_test_scaler.fit_transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_classifier.fit(x_scaled, y_train)

predict = knn_classifier.predict(x_test_scaled)

print("Actual vs Predicted: ")
print(np.concatenate((predict.reshape(len(predict), 1), y_test.reshape(len(y_test), 1)), 1))

# calculating the accuracy and the conf matrix
accuracy = accuracy_score(y_test, predict)
print("Accuracy: ", accuracy)

confusion_matrix = confusion_matrix(y_test, predict)
print("Confusion Matrix: ", confusion_matrix)