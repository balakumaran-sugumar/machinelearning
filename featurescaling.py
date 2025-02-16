import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def split_data():
    # read data in dataframe
    dataframe = pd.read_csv('iris.csv')

    # get X and Y dataset
    X = dataframe.drop('target', axis=1)
    y = dataframe['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("x_train: ", np.shape(X_train))
    print("x_test", np.shape(X_test))
    print("y_train", np.shape(y_train))
    print("y_test", np.shape(y_test))

    scaler = StandardScaler()
    x_train_fit = np.round(scaler.fit_transform(X_train), 5)
    x_test_fit = np.round(scaler.fit_transform(X_test), 5)

    print("x_train_fit: ", x_train_fit)
    print("x_test_fit: ", x_test_fit)


if __name__ == '__main__':
    split_data()