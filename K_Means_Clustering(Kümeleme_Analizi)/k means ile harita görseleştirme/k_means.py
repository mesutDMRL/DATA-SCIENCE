# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:45:16 2022

@author: mesut
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("train.csv")

df = df.drop(["Descript", "DayOfWeek", "PdDistrict", "Resolution", "Address"], axis = 1)

scaler = MinMaxScaler()

print(df.isnull().sum())

# sadece 2014 verisi ile çalışmak istiyoruz.
f = lambda x :(x["Dates"].split())[0]
df["Dates"] = df.apply(f, axis = 1)

f = lambda x:(x["Dates"].split("-"))[0]
df["Dates"] = df.apply(f, axis = 1)

df_2014 = df[(df["Dates"] == "2014")]


scaler = MinMaxScaler()

scaler.fit(df_2014[["X"]])
df_2014["X_scaled"] = scaler.transform(df_2014[["X"]])
# Değeleri 1 ve 0 arasına sıkıştırıyoruz.
scaler.fit(df_2014[["Y"]])
df_2014["Y_scaled"] = scaler.transform(df_2014[["Y"]])


# K Değerini Belirleyelim (Elbow Yöntemini Kullanarak)
k_range = range(1,15)
list_dist = []
for k in k_range:
    KMeans_Modelim = KMeans(n_clusters=k)
    KMeans_Modelim.fit(df_2014[["X_scaled", "Y_scaled"]])
    list_dist.append(KMeans_Modelim.inertia_)


plt.xlabel("K")
plt.ylabel("Distortion Değeri(inertia)")
plt.plot(k_range, list_dist)
plt.axvline(x=5, color='orange', linestyle='--')
plt.annotate('elbow point = 5', xy=(5.2,3000), color='red')
plt.show()

kmeans_model = KMeans(n_clusters=5)
y_predicted = kmeans_model.fit_predict(df_2014[["X_scaled", "Y_scaled"]])

df_2014["cluster"] = y_predicted

# for geographical map drawing we will use plotly library.
import plotly.express as px

# y enlem, x ise boylam
figure = px.scatter_mapbox(df_2014, lat = "Y", lon = "X",
                           center = dict(lat = 37.8, lon = -122.4), # sanfrancisco'nun koordinatları
                           zoom = 9,
                           opacity = .9,  # transparanlık > 0 ve 1 arasında değer alır
                           mapbox_style = "open-street-map",             #"stamen-terrain",
                           # open_street-map siyasi harita gibi, stamen-terrain coğrafi harita gibi
                           color = "cluster", # renklendirmeyi neye göre yapacağı
                           title = "San Francisco Crime Districts",
                           width = 1100,
                           height = 700,
                           hover_data = ["cluster", "Category", "Y", "X"] # noktaların detay bilgisinin içeriği
                           )
figure.show()

# haritamızı bir html dosyasına çevirip python dosyamızdan bağımsız olarak bir web sitesi olarak ulaşabiliriz.
# default olarak python dosyasının olduğu dizine kaydeder.
import plotly
plotly.offline.plot(figure, filename = "maptest.html", auto_open = True)

#base map(mapbox_style) değiştirmek istenirse veya diğer metodlar kullanılmak istenirse
help(px.scatter_mapbox)










