# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:11:05 2022

@author: mesut
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits

# digits = load_digits()


# %matplotlib inline
# import matplotlib.pyplot as plt

# # plt.gray() 
# # for i in range(4):
# #     plt.matshow(digits.images[i]) 


# df = pd.DataFrame(digits.data)
# df['target'] = digits.target

# X = df.drop('target',axis='columns')
# y = df.target


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=20, random_state = 42)
# model.fit(X_train, y_train)

# print(model.score(X_test, y_test))

# y_predicted = model.predict(X_test)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_predicted)

# import seaborn as sn
# plt.figure(figsize=(10,7))
# sn.heatmap(cm, annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')


from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data)
df["target"] = iris.target

x = df.drop("target", axis = 1)
y = df.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))

y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize =(10, 7))
sn.heatmap(cm, annot = True)
plt.xlabel("Predicted")
plt.ylabel("Truth")