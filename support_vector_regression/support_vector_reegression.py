import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing the SVR model
from sklearn.svm import SVR


def support_linear_regression():
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:-1].values  # taking on the position number
    y = dataset.iloc[:, -1].values

    # not going to split the data to leverage the entire data set
    # apply feature scaling
    sc_x = StandardScaler()
    X = sc_x.fit_transform(X)
    # need to feature scaling to Y to avoid being neglected by svr
    # and if the feature is lower than the target, svr might neglect the feature data
    # hence applying feature scaling to both target and the dependent variable
    # converting the array into a 2d array from [ ] to a column [
    #                                                              ]
    sc_y = StandardScaler()
    y_arr = np.array(y).reshape(-1, 1)  # 10 rows and 1 column
    y = sc_y.fit_transform(y_arr)
    # y = y.ravel()

    # training the svr model
    regressor = SVR(kernel='rbf')
    regressor.fit(X, y)

    y_pred = regressor.predict(sc_x.transform([[6.5]]))
    print("The predicted salary is for linear reg: ", sc_y.inverse_transform(np.array(y_pred).reshape(-1, 1)))

    # plot the data
    plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color='red')
    plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
    plt.title("Truth or Bluff (SVR regression)")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.savefig("svr_reg.png")


if __name__ == '__main__':
    support_linear_regression()
