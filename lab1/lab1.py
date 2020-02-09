import numpy as np
import pandas as pd  # To read data
import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression

df = pd.read_excel('Faults.xlsx')  # load data set
X = df.iloc[:, 14].values.reshape(-1, 1)  # values converts it into a numpy array
Y = df.iloc[:, 15].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()