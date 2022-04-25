# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:38:48 2022

@author: mesut
"""

# ileri düzey matematiksel işlemler yapabileceğimiz bir kütüphane
import numpy as np

a=[1,4,7]
b=[5,7,2]
# print(a*b) hata verir

a=np.array([1,4,7])
b=np.array([5,7,2])
# print(a*b)

a=np.array([4.3,6.3,7.5])

b=np.array([[1,2,3], [4,5,6]])
# print(b)

temp1 = b.ndim # dizinin kaç boyutlu olduğunu verir.
temp2 = a.ndim
# print(temp1, temp2)

temp3 = b.shape # satır ve sütun sayımızı verir.
temp4 = a.shape 
# print(temp3, temp4)

temp5 = b.dtype # dizinin tipini öğrenmek için dtype fonksiyonunu kullanırız.
temp6 = a.dtype
# print(temp5, temp6)

# print(b[1,1]) # dizinin belirli bir hücresine erişmek için satır ve sütun sayıları kullanılır

# ilk satırdaki ikinci ve üçüncü değere ulaşmak için
# print(b[0,1:]) # ilk satırın 1. değerden sonraki elemanlarını yazdırır.

b[1,1] = 18 # belirli bir hücreye ulaşıp üzerinde işlem yapmak mümkün.
# print(b[1,1])

dizi_0 = np.zeros((2,3)) # tüm parametreleri 0 olan 2 ye 3 lük bir matris tanımladık.
# print(dizi_s)

dizi_2 = np.full((4, 4), 50) # tüm elemanları 50 olan 4 e 4 lük bir dizi tanımladık.
# print(dizi_2)

dizi_3 = np.random.rand(4,3) # tüm elemanları 0 ve 1 arasında random değer alan 4 e 3 lük bir dizi tanımladık.
# print(dizi_3)

dizi_4 = np.random.randint(0,100,size=(5,5)) # tüm elemanları 0 ve 100 arasında random değerler alan 5 e 5 lik bir dizi tanımladık.
# print(dizi_4)

#  DİZİ KOPYALARKEN DİKKAT!!!

a = np.array([1,2,3])
b = a
b[0] = 90 # b bir pointer gibi a ile aynı memory adresini gösteriyor. O nedenle b' de ki değişiklikler a'da da oluyor.
# print(a) # bu durumdan kaçınmak için copy() fonksiyonu kullanılmalı.

c = np.array([1,2,3])
d = c.copy() # artık aynı adreste tutulmuyorlar. d'de ki değişiklik c'yi etkilemez.
d[0] = 90
# print(c) 


# DİZİLERDE MATEMATİKSEL İŞLEMLER


a = np.array([1,2,3])
# print(a + 4) # tüm elemanlara uygulanır.
# print(a*2)
# print(a/2)
# print(a**3)


# İSTATİSTİK


dizi_5 = np.array([[1,2,3], [4,5,6]])
temp7 = np.min(dizi_5) # dizinin minumum değerini verir.
# print(temp7)

temp8 = np.max(dizi_5) # dizinin maximum değerini verir.
# print(temp8)

temp9 = np.sum(dizi_5) # dizi elemanlarının toplamını verir.
# print(temp9)


# DOSYADAN DİZİ YÜKLEME


filedata = np.genfromtxt("ornek.txt", delimiter = ",") # aynı veriyolu üzerindeki ornek.txt dosyasını genfromtxt fonksiyonu ile içeri aktardık.
# ayraç olarak "," kullandığımızı belirttik.
filedata = filedata.astype("int32") # veri tipi olarak int'i belirledik.
# print(filedata)

# EXTRALAR

temp00 = np.arange(0,20,2) # 0 ve 2 arasında 2'şer artan bir dizi oluşturur.
print(temp00)

