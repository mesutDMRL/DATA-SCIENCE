# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:59:39 2022

@author: mesut
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df =pd.read_csv("Avm_Musterileri.csv")

# plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
# plt.xlabel("Annual Income")
# plt.ylabel("Spending Score")
# plt.show()

# Bazı sütunların ismi çok uzun. İsimlerini değiştirerek kısaltalım.
df.rename(columns = {"Annual Income (k$)": "income"}, inplace = True)
df.rename(columns = {"Spending Score (1-100)": "score"}, inplace = True)
# inplace = True > mevcut dataframe üzerinde değişiklik yapar.
# inplace = False > Değişikliğin yapıldığı yeni bir dataframe döndürür. Eşitşiğin diğer tarafında yeni bir dataframe tanımlamak gerekir.

# Normalization için MinMaxScaler() fonksiyonunu kullaacağız.
scaler = MinMaxScaler()

scaler.fit(df[["income"]])
df["income"] = scaler.transform(df[["income"]])
# Değeleri 1 ve 0 arasına sıkıştırıyoruz.
scaler.fit(df[["score"]])
df["score"] = scaler.transform(df[["score"]])

# K Değerini Belirleyelim (Elbow Yöntemini Kullanarak)
k_range = range(1,11)
list_dist = []
for k in k_range:
    KMeans_Modelim = KMeans(n_clusters=k)
    KMeans_Modelim.fit(df[["income", "score"]])
    list_dist.append(KMeans_Modelim.inertia_)

plt.xlabel("K")
plt.ylabel("Distortion Değeri(inertia)")
plt.plot(k_range, list_dist)
plt.axvline(x=5, color='orange', linestyle='--')
plt.annotate('elbow point = 5', xy=(5.2,15), color='red')
plt.show()

# En iyi k değeri 5 olarak bulundu.
# K = 5 için modelimizi oluşturalım.
kmeans_model = KMeans(n_clusters=5)
# kmeans_model.fit(df[["income", "score"]])
y_predicted = kmeans_model.fit_predict(df[["income", "score"]])

df["cluster"] = y_predicted

# centroidleri görelim:
center = kmeans_model.cluster_centers_

df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]
df3 = df[df.cluster == 3]
df4 = df[df.cluster == 4]

plt.xlabel("income")
plt.ylabel("score")

plt.scatter(df0.income, df0.score, color = "red")
plt.scatter(df1.income, df1.score, color = "blue")
plt.scatter(df2.income, df2.score, color = "orange")
plt.scatter(df3.income, df3.score, color = "purple")
plt.scatter(df4.income, df4.score, color = "green")

# center numpy 2 boyutlu olduğu için x ve y sütunlarını center[:,0] ve center[:,1] şeklinde scatter için alıyoruz.
plt.scatter(center[:,0], center[:,1], color = "black", label = "centroid")
plt.legend()
plt.show()


