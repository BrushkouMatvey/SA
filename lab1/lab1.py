import numpy as np
import pandas as pd  # To read data
import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression

data = pd.read_excel('Faults.xlsx')  # load data set

Edges_Index_X = data.iloc[:, 14].values.reshape(-1, 1)# values converts it into a numpy array
Empty_Index_Y = data.iloc[:, 15].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(Edges_Index_X, Empty_Index_Y)  # perform linear regression
Y_pred = linear_regressor.predict(Edges_Index_X)  # make predictions

plt.scatter(Edges_Index_X, Empty_Index_Y)
plt.plot(Edges_Index_X, Y_pred, color='red')
plt.xlabel('Edges_Index')
plt.ylabel('Empty_Index')

print(data.iloc[:, [14, 15]].corr())
plt.show()