from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data():
    data_set = pd.read_csv('Data.csv')

    # get the feature set
    X = data_set.iloc[:, :-1]
    # this is the target set
    y = data_set.iloc[:, -1]

    # apply OneHot encoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = ct.fit_transform(X)
    print(X)

    le = LabelEncoder()
    le = le.fit_transform(y)
    print(le)


if __name__ == '__main__':
    load_data()
