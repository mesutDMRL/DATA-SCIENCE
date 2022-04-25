# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:45:20 2022

@author: mesut
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

df = pd.read_csv("NLPlabeledData.tsv", delimiter = "\t", quoting=3)

def process(review):
    # HTML taglarından kurtulma
    review = BeautifulSoup(review).get_text()
    # noktalama işaretleri ve sayılardan kurtulma
    review = re.sub("[^a-zA-Z]", " ", review)
    # büyük harfleri küçük harf yapma
    review = review.lower()
    # split ile tüm kelimeleri ayırma
    review = review.split()
    # stopwords'lerden kurtulma (tek başına bir anlam ifade etmeyen kelimeler)
    swords = set(stopwords.words("english"))
    review = [w for w in review if w not in swords]
    # kelimeleri tekrar birleştirip return ediyoruz
    return(" ".join(review))

# ilerlemeyi görmek için her 1000 review'dan sonra ekrana yazı yazdırıyoruz.
train_x_tum = []
for r in range(len(df.review)):
    if (r+1)%1000 ==0:
        print("No of reviews processed =", r+1)
    train_x_tum.append(process(df.review[r]))


x = train_x_tum
y = np.array(df.sentiment)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=42)

# countvectorizer kullanarak en çok kullanılan 5000 kelimenin olduğu bir bag of words matrisi oluşturuyoruz.
vectorizer = CountVectorizer(max_features= 5000)

# train verilerimizi feature matrisine dönüştürüyoruz.
x_train = vectorizer.fit_transform(x_train)

# x_train'i array'a dönüştürüyoruz çünkü fit işlemi için array istiyor
x_train = x_train.toarray()

model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(x_train, y_train)

# test verilerimizi feature vector matrisine dönüştürüyoruz.
# yani aynı işlemi(bag of words matrisine dönüştürme) tekrarlıyoruz test datamız için.
testxx = vectorizer.transform(x_test)

testxx = testxx.toarray()

test_predict = model.predict(testxx)
dogruluk = roc_auc_score(y_test, test_predict)

print("Dogruluk Oranı: %", dogruluk * 100)





    