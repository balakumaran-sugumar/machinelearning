# The ANN model is done using tensor flow

# part 1 data preprocessing
# part 2 Building the ANN
# part 3 Training the ANN
# part 4 Making the predictions and evaluating the model

# part 1: Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

print(tf.__version__)

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#print(X)
#print(y)

# One hot encoding for gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#print(" Post one hot encoding ", X)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)


# split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train)
print(y_test)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Building a ANN - creating sequence of layers
ann = tf.keras.models.Sequential()

# adding the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))   # rectifier activation function

# adding another hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))   # rectifier activation function

# adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# training the ANN with the training data
# compiling the ANN - Stochastic gradient descent - on each iteration instead of whole batch
# if the classification is not binray it will be categorigal_crossentropy
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=100)
y_pred = ann.predict(X_test)

# returns the probability
isleaving = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print(isleaving)
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

y_pred = (y_pred > 0.5)

print("Actual vs Predicted: ")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# calculating the accuracy and the conf matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", confusion_matrix)


# Need to train and get more details about different predictions and examples
