# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:21:22 2022

@author: mesut
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# KENDİ VERİ SETİMİZİ OLUŞTURMAK İÇİN İDEAL BİR YÖNTEM.
gelirler = np.random.normal(5000, 2000, 1000) # ortalama, standart sapma, eleman sayısı(matriste olabilir.)
# aylık geliri ortalama 5000 tl olan, standart sapması 2000 tl olan 1000 kişinin olduğu bir veri setimiz olduğunu düşünelim.
# print(gelirler) # ortalama ve standart sapma girdiğimiz değerlere yakın çıkar.

temp00 = gelirler.var() # var fonksiyonu veri setinin varyansını verir.
# print(temp00)

temp01 = gelirler.std() # std fonksiyonu veri setinin standart sapmasını verir.
# print(temp01)

temp1 = np.mean(gelirler) # mean fonksiyonu veri setinin ortalamasını verir. tam olarak 5000 vermez, yakın bir değer verir.
# print(temp1)

plt.hist(gelirler, 100) # hist(histogram) fonksiyonu zorunlu parametre olarak 2 değer alır. kullanacağımız veri grubu ve histogramdaki sütun sayısı.
plt.show()

temp2 = np.median(gelirler) # veri setinin medyanını verir. 5000 tl ler civarı
# print(temp2) # medyan: küçükten büyüğe sıraladığımızda ortada kalan değer.

# Diyelim ki Bii Gates'ın geliri bizim veri setimize karıştı.
gelirler = np.append(gelirler, [100000000]) # append fonksiyonu ile Bill Gates'i veri setimize ekledik.

temp3 = np.mean(gelirler)
# print(temp3) # aritmetik ortalama 5000 tl lerden 100000 tl lere fırladı ama medyan hala 5000 civarında yani doğru sonuç veriyor.
# medyan ve aritmetik ortalama arasında absürt bir fark varsa. Yukarda olduğu gibi veri setimizde bazı aykırı değerler var demektir.
# bu sorunla karşılaşıldığında aykırı değerler veri setinden ayıklanmalı. ya da mean(ortalama) kullanılmayacak.

#  MODE(MOD) ALMA >> dizide en çok tekrar eden değer.

# KENDİ VERİ SETİMİZİ OLUŞTURMAK İÇİN İDEAL BİR YÖNTEM.
yaslar = np.random.randint(7,18, size = 300) # low değer, high  değer, size:eleman sayısı
# randint fonksiyonu belirtilen aralıkta random int(tam sayı) değerler oluşturur.
# yaşları 7 ile 18 arasında değişen 300 öğrencinin olduğu bir veri setimiz olduğunu düşünelim.
# print(yaslar)

temp4 = stats.mode(yaslar)
# print(temp4) # Örnek Çıktı: ModeResult(mode=array([10]), count=array([39])) >> en sık tekrar(mod):10, tekrar sayısı:39

temp5 = np.mean(yaslar)
temp6 = np.median(yaslar) 