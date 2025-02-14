# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# libraries for working with dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer


def print_hi(name):

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    data_frame = pd.read_csv('Data.csv')

    # extracting feature dataset
    x = data_frame.iloc[:, :-1].values
    # extracting the target dataset
    y = data_frame.iloc[:, -1].values
    # print(y)
    imputer.fit(x[:, 1:3])
    x[:, 1:3] = imputer.transform(x[:, 1:3])

    print(x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
