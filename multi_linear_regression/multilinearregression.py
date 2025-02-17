import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression


def multi_regression():
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Need to apply onehot replacement since States is word and not numerical
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X = ct.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # print(X_train)

    # implementing the multi regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    predict = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    # should be the same shape
    print(np.concatenate((predict.reshape(len(predict), 1), y_test.reshape(len(y_test), 1)), 1))

    # predict for California, for admin-130000, marketing spenc=300000 and state = CAL
    print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))


if __name__ == '__main__':
    multi_regression()
