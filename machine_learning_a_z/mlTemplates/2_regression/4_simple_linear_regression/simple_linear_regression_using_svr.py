# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = y_train.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)
y_test = y_test.reshape(-1, 1)
y_test = sc_y.fit_transform(y_test)

from sklearn.svm import SVR
regressor = SVR(kernel='linear')
regressor.fit(X_train, y_train)
# regressor.fit(X_test, y_test)

# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

# Visualising the SVR results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Exp(SVR)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Exp(SVR)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# # Visualising the SVR results (for higher resolution and smoother curve)
# X_grid = np.arange(min(X), max(X), 0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color='red')
# plt.plot(X_grid, regressor.predict(X_grid), color='blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()


# Fitting Simple Linear Regression to the Training set
# from sklearn.linear_model import LinearRegression
#
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
#
# # Visualising the Training set results
# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
#
# # Visualising the Test set results
# plt.scatter(X_test, y_test, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
