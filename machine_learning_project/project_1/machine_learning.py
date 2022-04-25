# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:00:33 2022

@author: mesut
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml # mnist datasetini yüklemek için gerekli
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mnist = fetch_openml("mnist_784")

print(mnist.data.shape)

#  Mnist veriseti içindeki rakam fotograflarını görmek için bir fonksiyon tanımlayalım.
def showimage(dframe, index):
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)
    
    plt.imshow(some_digit_image, cmap = "binary")
    plt.axis("off")
    plt.show()
    
# Örnek Kullanım:
showimage(mnist.data, 1)

# test ve train oranı 1/7 ve 6/7
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

print(type(train_img))

# Rakam tahminlerimizi check etmek için train_img dataframe'ini kopyalıyoruz, çünkü az sonra değişecek
test_img_copy = test_img.copy() 
showimage(test_img_copy, 1)

# Verilerimizi scale etmemiz gerekiyor.
# çünkü PCA scale edilmemiş verilerde hatalı sonuçlar verebiliyor.
scaler = StandardScaler()

# scaler'ı sadece training seti üzerinde fit yapmamız yeterli.
scaler.fit(train_img)

# ama transform işlemini hem training hem de test'e yapmamız gerekiyor.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# PCA işlemini uyguluyoruz
# variance'ın 95% oranında korunmasını istediğimizi belirtiyoruz.
pca = PCA(.95)

# PCA'i sadece training sete yapmamız yeterli
pca.fit(train_img)

# Bakalım 784 boyutu kaç boyuta düşürebilmiş(%95 variance'ı koruyarak)
print(pca.n_components_)

# Şimdi transform işlemiyle hem train hem de test veri setimizin boyutlarını 784'ten 327'ye düşürelim
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# 2. AŞAMA
# Logistic Regression modelimizi PCA işleminden geçirilmiş veri setimiz üzerine uygulayacağız.
# Default solver çok yavaş çalıştığı için daha hızlı olan "lbfgs" solver'ı seçerek logistikregression nesnemizi oluşturuyoruz.

logisticReg = LogisticRegression(solver = "lbfgs", max_iter=10000)

# Logistic regression modelimizi train datamızı kullanarak eğitiyoruz

# birkaç dk sürebelir
logisticReg.fit(train_img, train_lbl)

# Modelimiz eğitildi şimdi el yazısı rakamları makine öğrenmesi ile tanıma işlemini gerçekleştirelim.
logisticReg.predict(test_img[0].reshape(1,-1))
showimage(test_img_copy, 0)

# Modelimizin doğruluk oranını(accuracy) ölçmek
print(logisticReg.score(test_img, test_lbl))

















