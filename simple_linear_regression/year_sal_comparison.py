import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# for linear regression
from sklearn.linear_model import LinearRegression


def sal_prediction():
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # training the simple linear regression model on the training set
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)

    # predict the test set results
    prediction = linear_reg.predict(X_test)
    print("Prediction: \n", prediction)
    print("Actual: \n", y_test)

    print("Prediction of a 20 year old experience: ", linear_reg.predict([[20]]))
    print("The coeff (b0) and the intercept (b1)", linear_reg.coef_, linear_reg.intercept_)

    # Plot Training Set Results
    plt.scatter(X_train, y_train, color='red', label='Actual Data')
    plt.plot(X_train, linear_reg.predict(X_train), color='blue', label='Regression Line')
    plt.title('Salary Vs Experience (Training Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.savefig("Training Data")

    plt.clf()
    # Plot Test Set Results
    plt.scatter(X_test, y_test, color='red', label='Actual Data')
    plt.plot(X_train, linear_reg.predict(X_train), color='blue',
             label='Regression Line')  # Use training set regression line
    plt.title('Salary Vs Experience (Test Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.savefig("Test Data")


if __name__ == '__main__':
    sal_prediction()
