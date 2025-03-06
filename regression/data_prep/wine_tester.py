# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Wine Quality Red dataset
wine_dataset = pd.read_csv('winequality-red.csv')

# Separate features and target
X = wine_dataset.iloc[:, :-1] # all data except the last column
y = wine_dataset.iloc[:, -1]  # only the last column


# Split the dataset into an 80-20 training-test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create an instance of the StandardScaler class
sc = StandardScaler()

# applying the scaling to all the columns since they are not transformed to oneHotColumn transformation
# Fit the StandardScaler on the features from the training set and transform it
x_train[:] = sc.fit_transform(x_train[:])

# Apply the transform to the test set
x_test[:] = sc.fit_transform(x_test[:])

# Print the scaled training and test datasets
print(x_train)
print(x_test)