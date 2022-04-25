# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 20:33:30 2022

@author: mesut
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn .preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA #**************
from sklearn.preprocessing import StandardScaler


df= pd.read_csv("Housing.csv")


x = df.drop(["price"], axis = 1)

y = df.price

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

le = LabelEncoder()
x.mainroad = le.fit_transform(x.mainroad)
x.guestroom = le.fit_transform(x.guestroom)
x.basement = le.fit_transform(x.basement)
x.hotwaterheating = le.fit_transform(x.hotwaterheating)
x.airconditioning = le.fit_transform(x.airconditioning)
x.prefarea = le.fit_transform(x.prefarea)
x.furnishingstatus = le.fit_transform(x.furnishingstatus)

temp0= x.corr() # features'lar arasındaki korelasyon bilgisini verir.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

reg = LinearRegression()
reg.fit(x_train, y_train)

temp5=list(zip(x, reg.coef_))
# print("Intercept: ", reg.intercept_)
# print("Coefficients:")
# print(list(zip(x, reg.coef_)))


y_pred_reg= reg.predict(x_test)

reg_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_reg})


meanAbErr = metrics.mean_absolute_error(y_test, y_pred_reg)
print('R squared: {:.2f}'.format(reg.score(x,y)*100))
meanSqErr = metrics.mean_squared_error(y_test, y_pred_reg)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg))

# ******************************************************************************
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data = principalComponents, columns=["principal component 1", "principal component 2"])

final_dataframe = pd.concat([principalDF, df[["price"]]], axis = 1)

# plt.figure(figsize = (9,9))
# plt.xlabel("principal component 1")
# plt.ylabel("principal component 2")
# for i  in final_dataframe.index:
#     dftemp = final_dataframe[df.index == i]
#     plt.scatter(dftemp["principal component 1"], dftemp["principal component 2"], color = "red", s = 10)
# plt.savefig("İkinci_Grafiğim.png", dpi=300)
# plt.show()

# temp = pca.explained_variance_ratio_
# print(temp)
# temp2 = pca.explained_variance_ratio_.sum()
# print(temp2)