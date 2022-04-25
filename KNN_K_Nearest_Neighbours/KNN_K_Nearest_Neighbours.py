# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 22:34:57 2022

@author: mesut
"""

# ************************************SUPERVİSED LEARNING(DENETİMLİ ÖĞRENME)************************************

# İKİYE AYRILIR:
    # CLASSIFICATION(Sınıflandırma): Kategorik çıktılar üretir.
    # REGGESSION: Nümerik çıktılar üretir. 
    
# Classification Model:
# KNN - K Nearest Neighbours Modeli:
    # Algoritma, sınıfları belli olan bir örnek kümesindeki verilerden yararlanılarak kullanılmaktadır.
    # Örnek veri setine katılacak olan yeni verinin, mevcut verilere göre uzaklığı hesaplanıp,
    # k sayıda yakın komşuluğuna bakılır. Uzaklık hesapları için genelde 3 tip uzaklık fonksiyonu kullanılmaktadır:
        # Euclidean Uzaklık
        # Manhattan Uzaklık
        # Minkowski Uzaklık.(Default olarak seçili olan fonksiyondur.)
    # Dezavantajı: uzaklık hesabı yaparken bütün durumları sakladığından,
    # büyük veriler için kullanıldığında çok sayıda bellek alanına gereksinim duymaktadır.


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Outcome = 1 > şeker/diabet hastası,  0 > sağlıklı
data = pd.read_csv("diabetes.csv") # ayraç olarak ne kullandığımız bilgisini vermezsek default olarak "," olarak kabul eder.

seker_hastalari = data[data.Outcome == 1]
saglikli = data[data.Outcome == 0]

seker_hastalari2 = data[(data.Outcome == 1) & (data.Pregnancies == 2)]

# Şimdilik sadece gloucose'a bakarak örnek bir çizim yapalım.
plt.scatter(saglikli.Age, saglikli.Glucose, color = "green", label ="Sağlıklı", alpha = 0.4, s=40) 
#  scatter fonksiyonu grafiği noktalı olarak çizdirir. s = grafikteki noktaların boyutunu belirler.
# alpha = noktaların saydamlığını belirler.
plt.scatter(seker_hastalari.Age, seker_hastalari.Glucose, color = "red", label ="Diabet hastası", alpha=0.4, s=40)  
style.use('ggplot') # grafiğin arkaplanını ızgara şeklinde yapar.
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()

# Independent ve dependent variable'ları bulmak için alternatif yöntem. 
xx=data.iloc[:,0:8].values # Farkı burada geriye dataframe türünden değil array türünden döndürmesi.
yy=data.iloc[:,8:9].values

y = data.Outcome.values # Outcome verilerini tutar. dependent variable(bağımlı değişken)
x_ham_veri = data.drop(["Outcome"], axis = 1) # Outcome sütunun çıkarıldığı bir dataframe geriye döner.
# indipendent variables(bağımsız değişkenler)
# Bu şekilde bir  ayrıma gittik çünkü KNN algoritması x değerleri arasında sınıflandırma yapacak

# normalization yapıyoruz - x_ham_veri içerisindeki değerleri sadece 0 ve 1 arasında olacak şekilde hepsini güncelliyoruz.
# Eğer bu şekilde normalization yapmazsak yüksek sayılar küçük sayıları ezer ve KNN algoritmasını yanıltabilir.
x = (x_ham_veri - np.min(x_ham_veri)) / (np.max(x_ham_veri) - np.min(x_ham_veri))

# train datamız ile test datamızı ayırıyoruz.
# train datamız sistemin sağlıklı insan ile hasta insanı ayırt etmeyi öğrenmesi için kullanılıcak.
# test datamız ise modelimiz doğru bir şekilde hasta ve sağlıklı insanı ayırt edebiliyor mu? görmek için kullanılıcak.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10, random_state = 1)
# train_test_split fonksiyonu 4 değer return eder. Bu yüzden böyle bir kullanımı vardır.
# random_state parametresi her seferine aynı rastgeleliği sağlamak için kullanılır. verilen değer bir referans noktasıdır
# ve değer değişmediği sürece her aynı rastgele sayılar seçilir. 
# test_size parametresi verimizin ne kadarını test kümesi olarak ayıracağımızı belirlemek için kullanılır.
# test_size parametresi ile oynayıp daha yüksek doğruluk ile tahminler yapmak mümkün.

# KNN modelimizi oluşturuyoruz.
knn = KNeighborsClassifier(n_neighbors=3) # n_neighbors = k
knn.fit(x_train, y_train) # algoritmamızın kendini eğittiği kısım burasıdır.
prediction = knn.predict(x_test) # Şimdi x_test kümesindeki bağımsız değişkenler ile sağlıklı/sağlıksız tahmini yapıyor.
score = knn.score(x_test, y_test) # Tahmin sonuçlarını gerçek sonuçlarla karşılaştırıyoruz.
# print(score)

# k kaç olmalı?
# en iyi k değerini bir for döngüsü ile belirleyelim.

sayac = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    print("k = ", k, knn_yeni.score(x_test, y_test)) # en yüksek score k = 3 ve k = 6 için sağlandı.