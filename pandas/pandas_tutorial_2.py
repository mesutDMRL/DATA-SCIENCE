# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:03:30 2022

@author: mesut
"""

import pandas as pd

df=pd.read_csv("imdb_top_1000.csv")

df_kopya = df.copy() # copy fonksiyonu ile dataframe'mimizi kopyaladık.

temp1=df_kopya.head() # default olarak ilk 5 satırı sıralar.

# COLUMN DROP(SÜTUN SİLME)

df_yeni = df.drop(["Overview", "Meta_score"], axis = 1) # Overviev ve Meta_score sütunlarının olmadığı yeni bir dataframe döndürür.
# Yeni dataframe'i df_yeni ye eşitledik. axis = 1 bunun bir column drop işlemi olduğunu belirtir. axis = 0 row drop işlemi olduğunu belirtir.
df_yeni2 = df.drop(columns = ["Overview", "Meta_score"]) # axis'i kullanmadan bu şekilde de kullanılabilir.

# ROW DROP(SATIR SİLME)

df_yeni3 = df.drop([1]) # satırlarımızın ismi olmadığı için index numaralarını kullanıyoruz. 1. index'in olmadığı yeni bir dataframe tanımladık.
df_yeni4 = df.drop([1,2,3,4,5]) # birden fazla satırı silmekte mümkün.

# FİLTRELEME

df_filtered = df[df["IMDB_Rating"]>=9] # IMDB_Rating'i 9 dan büyük olan filmleri filtreledik. 
df_filtered2 = df.query("IMDB_Rating >= 9") # query fonksiyonu da aynı işi yapar.

# runtime sütun'u object türünden olduğu için üzerinde işlem yapamıyorum.
# öncelikle split metoduyla değerleri ayırıcam. örnek: 142 min >> [142] [min]
f = lambda x: (x["Runtime"].split())[0] # split default olarak boşluk alır.
df["RuntimeYeni"] = df.apply(f, axis=1) # apply fonksiyonu tüm dataframe'e uygulamamızı sağlar. 
# Dataframe'mimize RuntimeYeni adında yeni bir sütün ekledik.

df["RuntimeYeni2"] = df["RuntimeYeni"].astype("int") # yeni oluşturduğum sütunu object veri türünden int'e çevirdim.

temp10=df.sort_values("RuntimeYeni") # Runtime süresine göre sıraladım.
df_filtered3 = df.query("RuntimeYeni2 >= 140") 

df = df.drop(["RuntimeYeni"], axis = 1) # önceki column drop işleminden farkı yeni bir dataframe üzerinde işlem yapmadım.
# orjinal df dataframe'mimden sildim. drop fonksiyonu geriye dataframe türünden dönüştürdüğü için bu işlem mümkün.