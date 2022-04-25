# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:36:34 2022

@author: mesut
"""

import pandas as pd

df=pd.read_csv("imdb_top_1000.csv") # read_csv methodu aynı dosya yolu üzerindeki csv uzantılı dosyayı okumamızı sağlar.
# txt veya xls(excel) dosyaları da kullanabiliriz.

temp=df.head(5) # head fonksiyonu içine aldığı değer kadar satırı baştan itibaren döndürür. İlk 5 satır.
# print(temp) 

temp2=df.tail(5) # tail fonksiyonu içine aldığı değer kadar satırı sondan itibaren döndürür. Son 5 satır.
# print(temp2)

temp3=df.shape # shape fonksiyonu dataframe'in boyutunu döndürür.
# print(temp3)

temp4=df.columns # colums fonksiyonu sütun başlıklarını döndürür.
# print(temp4)

temp5=df.dtypes # dtypes fonksiyonu sütun başlıklarının veri tipini döndürür.
# print(temp5)

temp6=df.isnull() # verilerde eksik olan hücreleri döndürür.
# print(temp5)

temp7=df["Series_Title"] # bu şekilde sütun başlığını kullandığımızda sadece o sütunun altıdaki veriler sıralanır.
# print(temp7)

temp8=df["Series_Title"][:5] # ilgili sütunun altındaki ilk 5 satırı sıralar. 
# print(temp8)

temp9=df[["Series_Title", "Released_Year"]] # birden fazla sütünu da sıralamak mümkün.
# temp8 den farkı birden fazla sütun sıralandığında geriye döndürdüğü değer dataframe türünden olur.temp8 de series(seri) türünde bir değer döndürür.
# print(temp9)

temp10=df.sort_values("Released_Year") # seçilen başlık türünden sıralar. Bu örnekte Released_Year(Çıkış_Yılı) türünden.
# print(temp10)

temp11=df["Released_Year"].value_counts() # ilgili sütunda aynı değerlerin kaç defa tekrar ettiğini döndürür.
# print(temp11) # bu örnek özelinde hangi yıl kaç film çıktığını sıralar.


# ********************************VERİ FİLTRELEME***********************************


temp12=df["IMDB_Rating"][1] # index'i 1 olan IMDB_Rating sütundaki değeri döndürür.
print(temp12)

temp13=df.loc[df["Series_Title"]=="The Godfather"]["IMDB_Rating"] # Series_Title'ı The Godfather olan satırın IMDB_Rating özelliğini döndürür.
# print(temp13)

temp14=df.loc[df["Series_Title"]=="The Godfather"] # Series_Title'ı The Godfather olan satırı döndürür.
# print(temp14)

temp15=df.loc[(df["IMDB_Rating"]>8) & (df["No_of_Votes"]>=200000)] 
# IMDB_Rating'i 8 den büyük olan ve No_of_Votes(kullanıcı oyu) 200000 den fazla olan filmleri döndürür.
# print(temp15)

# Gross sütunu object veri türünden olduğu için üzerinde işlem yapamıyorum. Bunu düzeltmek için;
df["Gross"] = df["Gross"].str.replace(",", "") # öncelikle gross değerlerdeki virgüllerden kurtuluyorum.
df["Gross"] = pd.to_numeric(df["Gross"]) # gross sütununu object veri türünden numeric(int ya da float) veri türüne dönüştürdük.
# print(type(df["Gross"][0])) artık gross sütunundaki veriler üzerinden işlemler yapabilirim.

temp16=df.loc[(df["IMDB_Rating"]>8.1) & (df["Gross"]>=50000000)]
# print(temp16)

temp17=df.loc[1] # 1. satırı döndürdü.
# print(temp17)


# ********************************MANUEL OLARAK DATAFRAME OLUŞTURMA********************************


import random

randomlist1 = random.sample(range(15,25), 2) # 15 25 arasında 2 elemanı olan bir liste
randomlist2 = random.sample(range(15,25), 2)

randomListOfLists = [randomlist1, randomlist2]

columns1 = ["Sicaklik_1._Gun", "Sicaklik_2._Gun"]
mydataframe = pd.DataFrame(randomListOfLists, index=["Istanbul", "Ankara"], columns=columns1)
# Dataframe fonksiyonu ile manuel olarak değerlerini, index'ini ve sütunlarını vererek kendi dataframe'imizi oluşturuyoruz.
# print(mydataframe)
# print(type(mydataframe)) # veri türü dataframe olur.


temp00=mydataframe.iloc[0] # iloc fonksiyonunda index'in sırası kullanılır.
# print(temp00)
temp01=mydataframe.loc["Istanbul"] # loc fonksiyonunda index'in adı kullanılır. 
# temp00 ve temp01 aslında aynı satırı gösteriyor.
# print(temp01)
