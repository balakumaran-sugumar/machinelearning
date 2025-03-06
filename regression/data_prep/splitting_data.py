import pandas as pd
from sklearn.model_selection import train_test_split


def split_data():
    dataframe = pd.read_csv('Data.csv')
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print("x_train", x_train)
    print("x_test", x_test)
    print("y_train", y_train)
    print("y_test", y_test)


if __name__ == '__main__':
    split_data()