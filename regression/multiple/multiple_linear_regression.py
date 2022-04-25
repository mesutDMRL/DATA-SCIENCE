# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:18:19 2022

@author: mesut
"""
# Multiple liear regression formülü:
    # y = a + b1x1 + b2x2 + b3x3 + ....

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("multilinearregression.csv", sep=";")



reg = linear_model.LinearRegression()
reg.fit(df[["alan", "odasayisi", "binayasi"]], df["fiyat"])

# Modelimizin sonuçlarını değerlendirelim.
temp = reg.predict([[230,4,10]])
temp2 = reg.predict([[230,6,0]])
temp3 = reg.predict([[355,3,20]])
# veya
tempall = reg.predict([[230,4,10], [230,6,0], [355,3,20]])

temp4 = reg.coef_ # bağımsız değişkenlerin katsayısını verir.

a = reg.intercept_  # denklemdeki a değerini verir.

# sonuca denklem üzerinden bakıp. Modelin doğruluğunu test edelim.

b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230 
x2 = 4
x3 = 10

y = b1*x1 + b2*x2 + b3*x3 + a # temp ile aynı sonucu verdi.