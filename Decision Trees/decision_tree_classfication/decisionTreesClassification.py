# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 12:15:54 2022

@author: mesut
"""

import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")

# scikit-learn kütüphanesi decision tree'lerin düzgün çalışması için herşeyin rakamlsal olmasını bekliyor.
# Bu nedenle veri setimizdeki tüm Y ve N değerlerini 0 ve 1 olarak düzeltiyoruz.
# Aynı sebeple eğitim seviyesini de BS:0 MS:1 ve PhD:2 olarak güncelliyoruz.
# map() kullanarak boş hücreler veya geçersiz değer girilen hücreler NaN ile doldurulacaktır,
# buna şuandaki veri setimizde ihtiyacımız yok ama sizin ilerde yoğun veri ile çalıştığınız zaman ihtiyacınız olacaktır.

duzeltme_mapping = {"Y":1,"N":0}

df.IseAlindi = df.IseAlindi.map(duzeltme_mapping)
df["SuanCalisiyor?"] = df["SuanCalisiyor?"].map(duzeltme_mapping)
df["Top10 Universite?"] = df["Top10 Universite?"].map(duzeltme_mapping)
df["StajBizdeYaptimi?"] = df["StajBizdeYaptimi?"].map(duzeltme_mapping)
duzeltme_mapping_egitim = {"BS":0, "MS":1, "PhD":2}
df["Egitim Seviyesi"] = df["Egitim Seviyesi"].map(duzeltme_mapping_egitim)

y = df["IseAlindi"]
x = df.drop(["IseAlindi"], axis = 1)

# Decision Tree'mizi oluşturalım.
clf = tree.DecisionTreeClassifier()
clf2 = tree.DecisionTreeRegressor()
clf = clf.fit(x,y)

# Prediction yapalım.
# 5 yıl deneyimi olan, halihazırda bir yerde çalışan ve 3 eski şirkette çalışmış olan, eğitim seviyesi lisans,
# top-tier school mezunu değil.    
print (clf.predict([[5,1,3,0,0,0]]))

# Toplam 2 yıllık deneyim, 7 kez iş değiştirmiş, çok iyi bir okul mezunu, şuan çalışmıyor.
print(clf.predict([[2,0,7,0,1,0]]))