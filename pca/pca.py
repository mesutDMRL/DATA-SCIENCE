# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:25:39 2022

@author: mesut
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
df = pd.read_csv(url, names = ["sepal length", "sepal width", "petal length", "petal width", "target"])

features = ["sepal length", "sepal width", "petal length", "petal width"]
x = df[features]
y = df.target

# Değerleri scale(ölçek) etmemiz gerekiyor. Çünkü her bir features çok farklı boyutlarda ve bunların yapay zeka tarafından
# eşit ağırlıklarda dengelenmesi gerekiyor. Bu amaçla standart scaler kullanarak tüm verileri mean = 0 and variance = 1
# olacak şekilde değiştiriyoruz.
x = StandardScaler().fit_transform(x)
# Bu bir normalization işlemi değil. standart scale işlemi.

# Orjinal verimiz 4 boyuta sahip:"sepal length", "sepal width", "petal length", "petal width"
# Biz PCA yaparak bunları 2 boyuta indirgeyeceğiz ancak PCA indirgeme işlemi sonucunda elde edeceğimiz 2 boyutun herhangi
# bir anlam ifade etmeyen başlıkları olacak. Yani 4 Features'dan 2 tanesini basit bir şekilde atmak değil yaptığımız işlem.
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data = principalComponents, columns=["principal component 1", "principal component 2"])

# Şimdi target sütunumuzu da PCA dataframe'imizin sonuna ekleyelim.
final_dataframe = pd.concat([principalDF, df[["target"]]], axis = 1)
# concat fonksiyonu görüldüğü üzere iki dataframe'i birbirine bağlar.

# dataframe'imizi görselleştirip bakalım:
dfsetosa = final_dataframe[df.target == "Iris-setosa"]
#dfsetosa2 = final_dataframe[final_dataframe.target == "Iris-setosa"] # üstteki ile aynı anlama gelir.
dfvirginica = final_dataframe[df.target == "Iris-virginica"]
dfversicolor = final_dataframe[df.target == "Iris-versicolor"]

plt.xlabel("principal component 1")
plt.ylabel("principal component 2")

plt.scatter(dfsetosa["principal component 1"], dfsetosa["principal component 2"], color = "red")
plt.scatter(dfvirginica["principal component 1"], dfvirginica["principal component 2"], color = "blue")
plt.scatter(dfversicolor["principal component 1"], dfversicolor["principal component 2"], color =  "green")
plt.show()

# Daha profesyonel bir plotting yapalım:
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.title("profesyonel")
targets = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
colors = ["r", "b", "g"]
for tar, col in zip(targets, colors):
    dftemp = final_dataframe[df.target == tar]
    plt.scatter(dftemp["principal component 1"], dftemp["principal component 2"], color = col)
plt.show()
    
# veri setimin ne kadarını koruyabildiğimize bakalım.
temp = pca.explained_variance_ratio_# her sütunda veri setine ait bilginin ne kadarı olduğu yazıyor.
print(temp) # ilk sütunda yüze 72 civarı ikinci sütunda yüzde 23 civarı bir veri tutulmuş.
temp2 = pca.explained_variance_ratio_.sum() # bilginin ne kadarının korunduğunu verir.
print(temp2) # veri setimizin yüzde 95 ten fazlasını koruyabilmişiz.
    
    
    
     
   

