import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# importing the polynomial feature package
from sklearn.preprocessing import PolynomialFeatures


def polynomial_regression():
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:-1].values  # skipping the position column for simplicity
    y = dataset.iloc[:, -1].values

    regressor = LinearRegression()
    regressor.fit(X, y)

    # to mimic polynomial linear regression, create powers of features (independent variable)
    polynomial_reg = PolynomialFeatures(degree=4)
    x_poly = polynomial_reg.fit_transform(X)

    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_poly, y)

    # print(lin_reg_2.coef_)
    # print(lin_reg_2.intercept_)

    y_lin_pred = regressor.predict([[6.5]])
    print("The predicted salary is for linear reg: ", y_lin_pred)

    x_new = np.array([[6.5]])
    x_new_poly = polynomial_reg.fit_transform(x_new)
    y_pred = lin_reg_2.predict(x_new_poly)

    print("The predicted salary for Polynomial regression is: ", y_pred)

    plt.scatter(X, y, color='red')
    plt.plot(X, regressor.predict(X), color='blue')
    plt.title("Truth or Bluff (Linear Regression")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.savefig("linear_reg.png")

    plt.clf()
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg_2.predict(x_poly), color='blue')
    plt.title("Truth or Bluff (Polynomial Linear Regression")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.savefig("polynomial_linear_reg.png")


if __name__ == '__main__':
    polynomial_regression()
