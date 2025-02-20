import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def random_forest_regression():
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = RandomForestRegressor(random_state=0, n_estimators=10)
    regressor.fit(X, y)

    y_predict = regressor.predict([[6.5]])
    print(y_predict)

    # plot the data
    plt.scatter(X, y, color='red')
    plt.plot(X, regressor.predict(X), color='blue')
    plt.title("Truth or Bluff (Random Forest regression tree)")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.savefig("ensemble_decision_tree.png")

if __name__ == '__main__':
    random_forest_regression()