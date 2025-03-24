import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(x_scaled, y_train)

predict = svm_classifier.predict(x_test_scaled)

print("Actual vs Predicted: ")
print(np.concatenate((predict.reshape(len(predict), 1), y_test.reshape(len(y_test), 1)), 1))

# calculating the accuracy and the conf matrix
accuracy = accuracy_score(y_test, predict)
print("Accuracy: ", accuracy)

confusion_matrix = confusion_matrix(y_test, predict)
print("Confusion Matrix: ", confusion_matrix)

# applying K cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=svm_classifier, X=X_train, y=y_train, cv=10)
print("Accuracies:\n", accuracies.mean())
print("Accuracies:\n", accuracies.std())

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'kernel': ['rbf']}]

grid_search = GridSearchCV(estimator=svm_classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Score: ", best_accuracy)
print("Best Parameters: ", best_parameters)

