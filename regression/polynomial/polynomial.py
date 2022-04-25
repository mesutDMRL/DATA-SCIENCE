# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:23:49 2022

@author: mesut
"""

# Polynomial Linear Regression
# Formülü:
    # y = a + b1x + b2x^2 + b2x^3 + ...... + b(n) * x^(n)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("polynomial.csv", sep = ";")

# veri setimizi çizdirelim.
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

# Görüldüğü gibi veriler doğrusal bir yapıda dağılmıyor.
# linear regression uygularsak hatalı sonuçlar elde ederiz.

reg = LinearRegression()
reg.fit(df[["deneyim"]], df.maas)

plt.xlabel("Deneyim")
plt.ylabel("maas")
plt.scatter(df.deneyim, df.maas)

x_ekseni = df.deneyim
y_ekseni = reg.predict(df[["deneyim"]])
plt.plot(x_ekseni, y_ekseni, color = "green", label = "linear regression")
plt.show()

# bir adet polynomaial nesnesi oluşturması için PolynomialFeatures fonksiyonunu çağırıyoruz.
# Bu fonksiyonu çağırırken polinomun derecesini (n) belirliyoruz.
# n değeriyle oynayarak daha doğru sonuçlar elde etmeye çalışabiliriz.
polynomial_regression = PolynomialFeatures(degree = 5) # degree = n > polinomun derecesi. > y = a + b1x + b2x^2

x_polynomial = polynomial_regression.fit_transform(df[["deneyim"]])

reg = LinearRegression()
reg.fit(x_polynomial, df["maas"])

y_head = reg.predict(x_polynomial)
plt.plot(df.deneyim, y_head, color = "red", label = "polynomial regression")
plt.legend()
plt.scatter(df.deneyim, df.maas)
plt.show()

x_polynomial1 = polynomial_regression.fit_transform([[4.5]]) # parametre olarak 2 boyutlu bir array istediği için 2 parantez var.
print(reg.predict(x_polynomial1))
