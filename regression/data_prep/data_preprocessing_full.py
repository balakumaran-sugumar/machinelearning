from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse


def feature_scaling():
    np.set_printoptions(suppress=True, precision=5)

    # read data from CSV
    file_data = pd.read_csv('country_data.csv')

    # split the data to x and y
    x = file_data.iloc[:, :-1]
    y = file_data.iloc[:, -1]

    # print("Before imputing X data\n", x)

    # correct missing data
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    x.iloc[:, 1:3] = imputer.fit_transform(x.iloc[:, 1:3])
    # print("After imputing X data\n", x)

    # transform data for feature scaling
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    x = ct.fit_transform(x)
    if issparse(x):
        x = x.toarray()
    le = LabelEncoder()
    y = le.fit_transform(y)

    x = np.round(x, 2)
    y = np.round(y, 2)

    # split the data into x and y set - feature and target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    print("X train: \n", x_train)
    print("Y train: \n", y_train)

    sc = StandardScaler()

    # do not apply standardization on transformed column - like onehotdata (will lose interpretation)
    # apply feature scaling only on the numerical variable, training and the test set should use the same scaler
    x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
    x_test[:, 3:] = sc.fit_transform(x_test[:, 3:])
    print("Post feature scaling and standardization for trainging data \n: ", x_train)
    print("Post feature scaling and standardization for test data \n: ", x_test)


if __name__ == '__main__':
    feature_scaling()
