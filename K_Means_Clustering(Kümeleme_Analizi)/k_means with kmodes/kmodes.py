# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:04:48 2022

@author: mesut
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

df = pd.read_csv("segmentation_data.csv")

df_temp = df[["ID", "Income", "Age"]]

df.drop(["ID"], axis = 1, inplace = True)


scaler = MinMaxScaler()
scaler.fit(df[["Age"]])
df.Age = scaler.transform(df[["Age"]])

scaler.fit(df[["Income"]])
df.Income = scaler.transform(df[["Income"]])


# Kullanacağımız model float değerlerle çalıştığı için değerleri float'a çevireceğiz.
mark_array = df.values

mark_array[:,2] = mark_array[:,2].astype(float) 
mark_array[:,4] = mark_array[:,4].astype(float) 


kproto = KPrototypes(n_clusters=10, verbose=2, max_iter = 20)
clusters = kproto.fit_predict(mark_array, categorical = [0,1,3,5,6])
# categoric veri içeren sütunları categorical parametresi ile belirtiyorum.

cluster_list = []
for i in clusters:
    cluster_list.append(i)
df["cluster"] = cluster_list

df[["ID", "Income", "Age"]] = df_temp

# df[df["cluster"] == 0].head(10)


colors = ["green", "red", "grey", "orange", "yellow", "cyan", "magenta", "black", "purple", "blue"]
plt.figure(figsize = (10,10))
plt.xlabel("Age")
plt.ylabel("Income")
for i in range(10):
    df_n = df[df["cluster"] == i]
    plt.scatter(df_n.Age, df_n.Income, color = colors[i], alpha = 0.4)




