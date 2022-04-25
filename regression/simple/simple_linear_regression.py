# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:09:07 2022

@author: mesut
"""
# Simple linear regression formülü:
    # y = a + bx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

data = pd.read_csv("Advertising.csv")
df = data.copy()
# print(df.isnull().sum()) # veri setimizde eksik olup olmadığını gösterir.

x = df[["TV"]]  # 2 adet köşeli parantez kullanıldığında dataframe olarak return eder.
y = df["Sales"] # 1 adet köşeli parantez kullanıldığında series olarak return eder.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression() 
lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_test) 

lin_reg_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
print('R squared: {:.2f}'.format(lin_reg.score(x,y)*100))

# denklem
# print("Sales= " + str("%.2f" % lin_reg.intercept_) + "+TV*" + str("%.2f" % lin_reg.coef_[0]))

#rmse değeri
rmse = np.sqrt(mean_squared_error(y_pred, y_test))

temp = np.std(y_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, lin_reg.predict(x_train), color = 'blue')
plt.title('Sales vs TV')
plt.xlabel('TV')
plt.ylabel('sales')
plt.show()


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, lin_reg.predict(x_train), color = 'blue')
plt.title('Sales vs TV - Predict')
plt.xlabel('TV')
plt.ylabel('sales')
plt.show()