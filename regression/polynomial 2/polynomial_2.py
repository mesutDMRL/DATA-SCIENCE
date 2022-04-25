# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:17:42 2022

@author: mesut
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt



df = pd.read_csv("winequality-red.csv")

y = df.drop(["quality"], axis = 1)
x = df.quality

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = PolynomialFeatures(degree = 4)
y_ = model.fit_transform(y) 
y_test_ = model.fit_transform(y_test)

reg = LinearRegression()
reg.fit(y_, x)
predicted_data = reg.predict(y_test_)
predicted_data = np.round_(predicted_data)

print (mean_squared_error(x_test,predicted_data))
# print (predicted_data)

reg_diff = pd.DataFrame({'Actual value': x_test, 'Predicted value': predicted_data})

plt.scatter(df.quality, df["alcohol"], s=2)
plt.xlabel("quality")
plt.ylabel("indipendent")
plt.show()

