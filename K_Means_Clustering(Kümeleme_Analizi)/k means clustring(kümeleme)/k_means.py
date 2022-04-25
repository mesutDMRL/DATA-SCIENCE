# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:46:02 2022

@author: mesut
"""
#  https://www.youtube.com/watch?v=5aUPhfjcuiA&lc=Ugz8kqXmBsVz6HwnlAh4AaABAg.9YB8VyMRNho9YBHrdSrxoX&ab_channel=TirendazAkademi
# import warnings
# warnings.filterwarnings('ignore')
# import sys
# !{sys.executable} -m pip install mglearn
import mglearn # makine öğrenmesini öğretmek için geliştirilen bir paket
# mglearn.plots.plot_kmeans_algorithm()
# mglearn.plots.plot_kmeans_boundaries()

from sklearn.datasets import make_blobs
# 4 ayrı bölge içeren 2 boyutlu bir veriseti üretelim.
x,y = make_blobs(n_samples=300, centers=4, cluster_std = 0.6, random_state=0)

import matplotlib.pyplot as plt
%matplotlib inline
# plt.scatter(x[:,0], x[:,1], s=50)


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)
kmeans.fit(x)

y_kmeans=kmeans.predict(x)

# plt.scatter(x[:,0], x[:,1], s=50, c=y_kmeans, cmap="viridis")
# c argümanı etiketlenmiş verileri renklendirmeye yarar. cmap kümelerin renklerini belirlemeye yarar.
centers = kmeans.cluster_centers_
# plt.scatter(centers[:,0], centers[:,1], c="black", s=200, alpha=0.5)

from sklearn.datasets import make_moons
x1, y1 = make_moons(200, noise=.05, random_state=0)
labels = KMeans(2, random_state=0).fit_predict(x1)
# plt.scatter(x1[:,0], x1[:,1], s=50, c=labels, cmap="viridis")

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters = 2, affinity="nearest_neighbors", assign_labels="kmeans")
labels2 = model.fit_predict(x1)
# plt.scatter(x1[:,0], x1[:,1], s=50, c=labels2, cmap="viridis")

from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")

# ax = plt.axes(xticks=[], yticks=[])
# ax.imshow(china)
# print(china.shape)

veri = china/255
print(veri.shape)
veri = veri.reshape(427*640,3)
print(veri.shape)

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(veri)
temp = kmeans.fit(veri)

print(temp)
china_recolored = veri.reshape(china.shape)
fig, ax = plt.subplots(1, 2, figsize=(16, 6),
subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16);




