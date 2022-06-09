# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:41:05 2022

@author: mesut
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
listFPR = []
listTPR = []

df = pd.read_csv("Breast Cancer Wisconsin (Diagnostic) Data Set.csv")

df.drop(["Unnamed: 32", "id"], axis= 1, inplace = True)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df.diagnosis = le.fit_transform(df["diagnosis"])

x = df.drop(["diagnosis"], axis = 1)
y = df["diagnosis"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state = 0)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred), "DT")
cm = confusion_matrix(y_test, y_pred)
liste_prevalence = []
liste_prevalence.append(cm)
tprDT = cm[1,1]/(cm[1,0] + cm[1,1])
fprDT = cm[0,1]/(cm[0,0] + cm[0,1])
listTPR.append(tprDT*100)
listFPR.append(fprDT*100)
probs = dtc.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title("ROC")
plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
plt.legend(loc = "lower right")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.xlim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred), "RF")
cm = confusion_matrix(y_test, y_pred)
liste_prevalence.append(cm)
tprRF = cm[1,1]/(cm[1,0] + cm[1,1])
fprRF = cm[0,1]/(cm[0,0] + cm[0,1])
listTPR.append(tprRF*100)
listFPR.append(fprRF*100)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred), "NB")
cm = confusion_matrix(y_test, y_pred)
liste_prevalence.append(cm)
tprNB = cm[1,1]/(cm[1,0] + cm[1,1])
fprNB = cm[0,1]/(cm[0,0] + cm[0,1])
listTPR.append(tprNB*100)
listFPR.append(fprNB*100)

from sklearn.svm import SVC
svc = SVC(kernel = "rbf")
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred), "SVM")
cm = confusion_matrix(y_test, y_pred)
liste_prevalence.append(cm)
tprSVM = cm[1,1]/(cm[1,0] + cm[1,1])
fprSVM = cm[0,1]/(cm[0,0] + cm[0,1])
listTPR.append(tprSVM*100)
listFPR.append(fprSVM*100)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(x_train, y_train)
y_pred = logr.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred), "LR")
cm = confusion_matrix(y_test, y_pred)
liste_prevalence.append(cm)
tprLR = cm[1,1]/(cm[1,0] + cm[1,1])
fprLR = cm[0,1]/(cm[0,0] + cm[0,1])
listTPR.append(tprLR*100)
listFPR.append(fprLR*100)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric = "minkowski")
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred), "KNN")
cm = confusion_matrix(y_test, y_pred)
liste_prevalence.append(cm)
tprKNN = cm[1,1]/(cm[1,0] + cm[1,1])
fprKNN = cm[0,1]/(cm[0,0] + cm[0,1])
listTPR.append(tprKNN*100)
listFPR.append(fprKNN*100)

listef = []
listef.append(fprSVM*100)
listef.append(fprLR*100)
listef.append(fprDT*100)
listet = []
listet.append(tprSVM*100)
listet.append(tprLR*100)
listet.append(tprDT*100)
plt.scatter(listFPR, listTPR)
plt.plot(listef, listet)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.ylim([75,100])
plt.xlim([0,15])
plt.show()
# We can visualize the tree diagram of this model using Graphviz.
# from sklearn.tree import export_graphviz
# import graphviz
# columns = x.columns
# dot_data = export_graphviz(dtc,feature_names=columns, filled=True, rounded=True)

# graph = graphviz.Source(dot_data)
# graph.render("tree") 
