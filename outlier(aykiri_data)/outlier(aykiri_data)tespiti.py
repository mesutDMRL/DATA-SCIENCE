# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:18:07 2022

@author: mesut
"""

# **************************OUTLIER(AYKIRI DEĞER) TESPİTİ VE VERİSETİNDEN TEMİZLENMESİ**************************

# Outlier: Bir veriseti içerisinde diğer gözlemlerden aykırı ve sapan gözlem veya veri değeridir.

# Outlier Nedenleri:
    # İnsan kaynaklı hatalar(hatalı veri girişi).
    # Cihazlardan kaynaklanan hatalar(ölçüm cihazlarlarının nadiren de olsa yanlış değer ölçmesi).

# Bunu çözmek için verisetimizin alt ve üst limitini belirleyip bu sınıflar dışında kalan değerleri(outlier) verisetimizden çıkarıcaz.


import pandas as pd

df = pd.read_csv("outlier_ornek_veriseti.csv", sep=";")

describe = df.describe() # describe fonksiyonu veri setine ilişkin istatistiksel veriler verir.
# count: veri sayısı, mean: aritmetik ortalama, std: standart sapma, min ve max değerler, %50: ortanca değer(medyan),
# %25(Q1): ilk yarının ortalaması, %75(Q3): ikinci yarının ortalaması. 
# describe fonksiyonu geriye dataframe türünden döndürür.

# Q1 ve Q3'ü hesaplamak için alternatif yöntem.

Q11 = describe.loc["25%"]["boy"]
Q33 = describe.loc["75%"]["boy"] 
IQR = Q33 - Q11


# Q1 (%25 PERCENTILE HESAPLAMA)

Q1 = df.boy.quantile(0.25) # quantile fonksiyonun içine 0.25 değerini verirsek ilk yarının medyanını verir.
# print(Q1) # df.boy' daki boy sütun ismi olan boy herhangi bir fonksiyon değil.

# Q3 (%75 PERCENTİLE HESAPLAMA)

Q3 = df.boy.quantile(0.75) # quantile fonksiyonun içine 0.75 değerini verirsek ikinci yarının medyanını verir.
# print(Q3)

medyan = df.boy.quantile(0.50) # bu şekilde medyanı(Q2) hesaplamakta mümkün.

IQR_degeri = Q3 - Q1

# ALT LİMİT VE ÜST LİMİT HESAPLAMA 

alt_limit = Q1 - 1.5 * IQR_degeri

ust_limit = Q3 + 1.5 * IQR_degeri

# ALT VE ÜST LİMİTİN DIŞINDA KALAN(Outlier) DEĞERLERİ FİLTRELEME İLE TESPİT EDECEĞİZ

temp1 = df[(df.boy < alt_limit) | (df.boy > ust_limit)] # | >> bu işaret veya anlamına geliyor. 

# Şimdi tespit ettiğimiz outlier değerleri verisetimizden çıkarıcaz.

df_outlier_filtrelenmis = df[(df.boy > alt_limit) & (df.boy < ust_limit)] # outlier değerlerin filtrelendiği yeni bir dataframe tanımlıyoruz.
# print(df_outlier_filtrelenmis)
