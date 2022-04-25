# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:31:59 2022

@author: mesut
"""

#****************************************RECOMMENDATİON SYSTEMS****************************************

import numpy as np
import pandas as pd

column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("users.data", sep="\t", names = column_names)

# kaç kayıt varmış görelim
print(len(df))

movie_titles = pd.read_csv("movie_id_titles.csv")

# kaç kayıt varmış görelim
print(len(movie_titles))

df = pd.merge(df, movie_titles, on = "item_id")

# timastamp sütununa ihtiyacımız yok silelim.
df = df.drop(["timestamp"], axis = 1)

# RECOMMENDATION SİSTEMİMİZİ KURUYORUZ.
# Öncelikle excel'deki pivot tablo benzeri bir yapı kuruyoruz.
# Bu yapıya göre her satır bir kullanıcı olacak şekilde (yani dataframem'imimiz index'i user_id olacak)
# Sütunlarda film isimleri (yani title sütunu) olacak,
# tablo içinde de rating değerleri olacak şekilde bir dataframe oluşturuyoruz.

moviemat = df.pivot_table(index= "user_id", columns = "title", values = "rating")

starwars_user_rating = moviemat["Star Wars (1977)"]

# AMAÇ: STAR WARS FİLMİNE BENZER FİLM ÖNERİLERİ YAPMAK
# corrwith() metodunu kullanarak Star Wars filmi ile korelasyonları hesaplatalım.
similar_to_starwars = moviemat.corrwith(starwars_user_rating)

# bazı kayıtlarda boşluklar olduğu için hata veriyor. similar_to_starwars bir seri, biz bunu corr_starwars isimli
# bir dataframe'e dönüştürelim ve NaN kayıtları temizleyip bakalım:
corr_starwars = pd.DataFrame(similar_to_starwars, columns = ["Correlation"])
corr_starwars.dropna(inplace = True)

# Elde ettiğimiz dataframe'i sıralayalım ve star wars'a en yakın tavsiye edeceği filmleri görelim. 
corr_starwars2 = corr_starwars.sort_values("Correlation", ascending=False)

# Görüldüğü gibi alakasız sonuçlar çıktı, bunun nedenini bu filmlerin çok az oy alması.
# Pek bilinmeyen bir filme 5 yıldız veren biri aynı zamanda star wars'a 5 yıldız verince algoritma bu iki film 
# arasında yüzde yüz korelasyon olduğunu düşünüyor. Yani az bilinen filmler outlier gibi davranıyor.
# Bu durumu düzeltmek için 100'den az oy alan filmleri eleyelim. Bu amaçla ratings
# isimli bir dataframe oluşturalım ve burada her filmin kaç tane oy aldığını tutalım.

# Her filmin ortalama(mean value) rating değerini bulalım.
ratings = pd.DataFrame(df.groupby("title")["rating"].mean())

# Şimdi her filmin aldığı oy sayısını bulalım.
ratings["rating_oy_sayisi"] = pd.DataFrame(df.groupby("title")["rating"].count())

# Büyükten küçüğe sıralayıp bakalım.
ratinga_to_sort = ratings.sort_values("rating_oy_sayisi", ascending=False)

# Tekrar esas amacımıza dönelim ve corr_starwars dataframe'imize rating_oy_sayisi sütununu ekleyelim.(join ile)
corr_starwars = corr_starwars.join(ratings["rating_oy_sayisi"])

# 100 den az oy alanları filtreleyelim.
corr_starwars3 = corr_starwars[corr_starwars["rating_oy_sayisi"]>=100].sort_values("Correlation", ascending = False)














