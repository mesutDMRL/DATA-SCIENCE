# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:24:34 2022

@author: mesut
"""
# https://towardsdatascience.com/train-a-regression-model-using-a-decision-tree-70012c22bcc1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

df = pd.read_csv("cali_housing.csv")

# sns.scatterplot(x = df.Longitude, y = df.MedHouseVal)

from sklearn.tree import DecisionTreeRegressor
X = df[['Longitude']] #Two-dimensional (pd DataFrame)
y = df['MedHouseVal'] #One-dimensional (pd Series)

#Create an object (model)
dtr1 = DecisionTreeRegressor(max_depth=7, random_state=1)

#Fit (train) the model
dtr1.fit(X, y)

# sns.scatterplot(x=df['Longitude'], y=df['MedHouseVal'], label='data')

# plt.plot(df['Longitude'].sort_values(), dtr1.predict(df['Longitude'].sort_values().to_frame()),
#           color='red', label='model', linewidth=2)
# plt.legend()

# We can visualize the tree diagram of this model using Graphviz.
from sklearn.tree import export_graphviz
import graphviz

# dot_data = export_graphviz(dtr1, feature_names=['Longitude'], filled=True, rounded=True)

# graph = graphviz.Source(dot_data)
# graph.render("tree") 

# max_depth=2 olduğu bu durumda, model eğitim verilerine çok iyi uymuyor. Buna yetersiz uyum sorunu denir.
# max_depth=15 olduğu bu durumda, model eğitim verilerine çok iyi uyuyor ancak
# yeni girdi verileri için genelleme yapamıyor. Buna aşırı uyum sorunu denir.
# Burada ağaç diyagramını görselleştiremiyoruz çünkü 32768 (2¹⁵) yaprak düğümü var!
# Burada model, verilerin gürültüsüne uyum sağlamıştır ve herhangi bir örüntü öğrenmek yerine verileri ezberlemeye çalışır.
# Bu nedenle, en iyi modeli oluştururken hem eksik hem de fazla uydurma koşullarından kaçınmalıyız.


# Peki, maksimum_derinlik hiperparametresi için en iyi değer nedir?
# max_depth için optimum değeri (çok küçük veya çok büyük değil) bulmaya hiperparametre ayarlama denir.

# KARAR AĞACI REGRESYONU İÇİN HİPERPARAMETRE  AYARLAMA
# İki Yöntem Var:
    # 1. Using Scikit-learn train_test_split() function
    # 2. Using k-fold cross-validation 
# ****Using Scikit-learn train_test_split() function:
from sklearn.model_selection import train_test_split
x= df.drop(["Longitude"], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42, shuffle = True)

from sklearn.metrics import mean_squared_error as mse

max_depth_list = range(1,20)

training_error = []
for max_depth in max_depth_list:
    model_1 = DecisionTreeRegressor(max_depth=max_depth)
    model_1.fit(X,y)
    training_error.append(mse(y, model_1.predict(X)))

testing_error = []
for max_depth in max_depth_list:
    model_2 = DecisionTreeRegressor(max_depth=max_depth)
    model_2.fit(x_train,y_train)
    testing_error.append(mse(y_test, model_2.predict(x_test)))
    
plt.plot(max_depth_list, training_error, color='blue', label='Training error')
plt.plot(max_depth_list, testing_error, color='green', label='Testing error')
plt.xlabel('Tree depth')
plt.axvline(x=7.5, color='orange', linestyle='--')
plt.annotate('optimum = 7.5', xy=(7.5, 1.17), color='red')
plt.ylabel('Mean squared error')
plt.title('Hyperparameter Tuning', pad=15, size=15)
plt.legend()
plt.show()

# Ağaç derinliği = 7 noktasında eğitim hatası sürekli azalmasına rağmen test hatası artmaya başlar. 
# Bu çizimden , maksimum_derinlik hiperparametresi için optimum değerin 7 olduğunu onaylayabiliriz.

# Karar ağaçlarının avantajları:
    # Karar ağaçlarının avantajları
    # Özellik ölçeklendirme gerektirmez
    # Doğrusal olmayan veriler için kullanılabilir
    # Parametrik olmayan: Verilerde çok az sayıda temel varsayım
    # Hem regresyon hem de sınıflandırma için kullanılabilir
    # Görselleştirmesi kolay
    # Yorumlanması kolay
# Karar ağaçlarının dezavantajları:
    # Karar ağacı eğitimi, özellikle model hiperparametresini k kat çapraz doğrulama yoluyla ayarlarken, 
    # hesaplama açısından pahalıdır.
    # Verilerdeki küçük bir değişiklik, karar ağacının yapısında büyük bir değişikliğe neden olabilir.
