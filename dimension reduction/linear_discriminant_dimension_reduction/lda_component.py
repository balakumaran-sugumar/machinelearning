import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# read data from CSV
file_data = pd.read_csv('wine.csv')

# split the data to x and y
x = file_data.iloc[:, :-1]
y = file_data.iloc[:, -1]

# split the data into x and y set - feature and target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

print("X train: \n", x_train)
print("Y train: \n", y_train)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# implementation of the PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
lda = lda(n_components=2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

print(x_train)

classfier = LogisticRegression(random_state=0)
classfier.fit(x_train, y_train)

y_pred = classfier.predict(x_test)

accuracy_score = accuracy_score(y_pred, y_test)
print(accuracy_score)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)