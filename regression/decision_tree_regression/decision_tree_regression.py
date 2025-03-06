import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def decision_tree_regression():
  # print("Decision tree regression")
  # not well adopted with single data set

  dataset = pd.read_csv('Position_Salaries.csv')
  X = dataset.iloc[:, 1:-1].values
  y = dataset.iloc[:, -1].values

  # training on the entire data set
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  # print(X_train)
  regressor = DecisionTreeRegressor(random_state=0)
  regressor.fit(X, y)

  y_predict = regressor.predict([[7]])
  print(y_predict)

  # plot the data
  plt.scatter(X, y, color='red')
  plt.plot(X, regressor.predict(X), color='blue')
  plt.title("Truth or Bluff (Decision tree regression)")
  plt.xlabel("Position Level")
  plt.ylabel("Salary")
  plt.savefig("decision_tree.png")

if __name__ == '__main__':
    decision_tree_regression()
