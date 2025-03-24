import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = np.where(y == 2, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)
#
# # scaling the feature set
# X_train_scaler = StandardScaler()
# x_scaled = X_train_scaler.fit_transform(X_train)
#
# X_test_scaler = StandardScaler()
# x_test_scaled = X_test_scaler.fit_transform(X_test)

predict = classifier.predict(X_test)

print("Actual vs Predicted: ")
print(np.concatenate((predict.reshape(len(predict), 1), y_test.reshape(len(y_test), 1)), 1))

# calculating the accuracy and the conf matrix
accuracy = accuracy_score(y_test, predict)
print("Accuracy: ", accuracy)

confusion_matrix = confusion_matrix(y_test, predict)
print("Confusion Matrix: ", confusion_matrix)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
