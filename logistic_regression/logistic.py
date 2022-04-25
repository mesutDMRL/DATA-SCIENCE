# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:31:50 2022

@author: mesut
"""

# https://www.kaggle.com/prashant111/logistic-regression-classifier-tutorial/notebook

import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

df = pd.read_csv("weatherAUS.csv")

col_names = df.columns

df.drop(["RISK_MM"], axis=1, inplace = True)

# find categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O'] # sıfır değil büyük O harfi.

# view the categorical variables
# print(df[categorical].head())

# check missing values in categorical variables
# print(df[categorical].isnull().sum())

# print categorical variables containing missing values
cat1 = [var for var in categorical if df[var].isnull().sum!=0]
# print(df[cat1].isnull().sum())

# view frequency of categorical variables
# for var in categorical: 
#     print(df[var].value_counts())

# view frequency distribution of categorical variables
# for var in categorical: 
#     print(df[var].value_counts()/np.float(len(df)))

# Kategorik bir değişken içindeki etiket sayısı, kardinalite olarak bilinir. 
# Bir değişken içindeki çok sayıda etiket, yüksek kardinalite olarak bilinir. 
# Yüksek kardinalite, makine öğrenimi modelinde bazı ciddi sorunlar doğurabilir. 
# Bu nedenle, yüksek kardinaliteyi kontrol edeceğim.

# check for cardinality in categorical variables
# for var in categorical:
#     print(var, ' contains ', len(df[var].unique()), ' labels')

# We can see that there is a Date variable which needs to be preprocessed. 
# I will do preprocessing in the following section.
# All the other variables contain relatively smaller number of variables.


# Feature Engineering of Date Variable
# print(df['Date'].dtypes)
# parse the dates, currently coded as strings, into datetime format
df.Date = pd.to_datetime(df.Date)

# extract year from date
df['Year'] = df['Date'].dt.year
# extract month from date
df['Month'] = df['Date'].dt.month
# extract day from date
df['Day'] = df['Date'].dt.day

# drop the original Date variable
df.drop('Date', axis=1, inplace = True)

# find categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O']

# check for missing values in categorical variables 
# print(df[categorical].isnull().sum())

# print number of labels in Location variable
# print('Location contains', len(df.Location.unique()), 'labels')

# check labels in location variable
# print(df.Location.unique())

# check frequency distribution of values in Location variable
# print(df.Location.value_counts())

# preview the dataset with head() method
# print(pd.get_dummies(df.Location, drop_first=True).head())

# print number of labels in WindGustDir variable
# print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')

# check labels in WindGustDir variable
# print(df['WindGustDir'].unique())

# check frequency distribution of values in WindGustDir variable
# print(df.WindGustDir.value_counts())

# print(pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head())
# print(pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0))

# find numerical variables
numerical = [var for var in df.columns if df[var].dtype!='O']

# view the numerical variables
# print(df[numerical].head())

# check missing values in numerical variables
# print(df[numerical].isnull().sum())

# view summary statistics in numerical variables
# print(round(df[numerical].describe()),2)

# draw boxplots to visualize outliers
# plt.figure(figsize=(15,10))

# plt.subplot(2, 2, 1)
# fig = df.boxplot(column='Rainfall')
# fig.set_title('')
# fig.set_ylabel('Rainfall')


# plt.subplot(2, 2, 2)
# fig = df.boxplot(column='Evaporation')
# fig.set_title('')
# fig.set_ylabel('Evaporation')


# plt.subplot(2, 2, 3)
# fig = df.boxplot(column='WindSpeed9am')
# fig.set_title('')
# fig.set_ylabel('WindSpeed9am')


# plt.subplot(2, 2, 4)
# fig = df.boxplot(column='WindSpeed3pm')
# fig.set_title('')
# fig.set_ylabel('WindSpeed3pm')


# plot histogram to check distribution
# plt.figure(figsize=(15,10))

# plt.subplot(2, 2, 1)
# fig = df.Rainfall.hist(bins=10)
# fig.set_xlabel('Rainfall')
# fig.set_ylabel('RainTomorrow')


# plt.subplot(2, 2, 2)
# fig = df.Evaporation.hist(bins=10)
# fig.set_xlabel('Evaporation')
# fig.set_ylabel('RainTomorrow')


# plt.subplot(2, 2, 3)
# fig = df.WindSpeed9am.hist(bins=10)
# fig.set_xlabel('WindSpeed9am')
# fig.set_ylabel('RainTomorrow')


# plt.subplot(2, 2, 4)
# fig = df.WindSpeed3pm.hist(bins=10)
# fig.set_xlabel('WindSpeed3pm')
# fig.set_ylabel('RainTomorrow')

# find outliers for Rainfall variable
# IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
# Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
# Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
# print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for Evaporation variable
# IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
# Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
# Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
# print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for WindSpeed9am variable
# IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
# Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
# Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
# print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

# find outliers for WindSpeed3pm variable
# IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
# Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
# Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
# print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)
     
for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 2.4)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)

cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression

# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)

# fit the model
y_train.fillna(y_train.mode()[0], inplace=True)
y_test.fillna(y_train.mode()[0], inplace=True)
logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test)


# print(logreg.predict_proba(X_test)[:,0])
# print(logreg.predict_proba(X_test)[:,1])


from sklearn.metrics import accuracy_score
# print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

# y_pred_train = logreg.predict(X_train)
# print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# # fit the Logsitic Regression model with C=100

# # instantiate the model
# logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# # fit the model
# logreg100.fit(X_train, y_train)

# # print the scores on training and test set

# print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

# print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))

# # fit the Logsitic Regression model with C=001

# # instantiate the model
# logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# # fit the model
# logreg001.fit(X_train, y_train)

# # print the scores on training and test set

# print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

# print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))


# check class distribution in test set
# print(y_test.value_counts())

# check null accuracy score
null_accuracy = (22067/(22067+6372))
# print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

# print('Confusion matrix\n\n', cm)

# print('\nTrue Positives(TP) = ', cm[0,0])

# print('\nTrue Negatives(TN) = ', cm[1,1])

# print('\nFalse Positives(FP) = ', cm[0,1])

# print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
#                                  index=['Predict Positive:1', 'Predict Negative:0'])
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred_test))

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# print classification accuracy
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
# print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# print classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
# print('Classification error : {0:0.4f}'.format(classification_error))

# print precision score
precision = TP / float(TP + FP)
# print('Precision : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)
# print('Recall or Sensitivity : {0:0.4f}'.format(recall))


true_positive_rate = TP / float(TP + FN)
# print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

false_positive_rate = FP / float(FP + TN)
# print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

specificity = TN / (TN + FP)
# print('Specificity : {0:0.4f}'.format(specificity))


# print the first 10 predicted probabilities of two classes- 0 and 1
y_pred_prob = logreg.predict_proba(X_test)[0:10]
# print(y_pred_prob)

# store the probabilities in dataframe
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])
# print(y_pred_prob_df)

# print the first 10 predicted probabilities for class 1 - Probability of rain
# print(logreg.predict_proba(X_test)[0:10, 1])

# store the predicted probabilities for class 1 - Probability of rain
y_pred1 = logreg.predict_proba(X_test)[:, 1]

# # plot histogram of predicted probabilities
# # adjust the font size 
# plt.rcParams['font.size'] = 12
# # plot histogram with 10 bins
# plt.hist(y_pred1, bins = 10)
# # set the title of predicted probabilities
# plt.title('Histogram of predicted probabilities of rain')
# # set the x-axis limit
# plt.xlim(0,1)
# # set the title
# plt.xlabel('Predicted probabilities of rain')
# plt.ylabel('Frequency')


from sklearn.preprocessing import binarize

# for i in range(1,5):
    
#     cm1=0
    
#     y_pred1 = logreg.predict_proba(X_test)[:,1]
    
#     y_pred1 = y_pred1.reshape(-1,1)
    
#     y_pred2 = binarize(y_pred1, i/10)
    
#     y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
#     cm1 = confusion_matrix(y_test, y_pred2)
#     print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
#             'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
#             cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
#             cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
#            'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
#            'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
#            'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
#             '====================================================', '\n\n')



# plot ROC Curve
from sklearn.metrics import roc_curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

# plt.figure(figsize=(6,4))

# plt.plot(fpr, tpr, linewidth=2)

# plt.plot([0,1], [0,1], 'k--' )

# plt.rcParams['font.size'] = 12

# plt.title('ROC curve for RainTomorrow classifier')

# plt.xlabel('False Positive Rate (1 - Specificity)')

# plt.ylabel('True Positive Rate (Sensitivity)')

# plt.show()


# compute ROC AUC
from sklearn.metrics import roc_auc_score

# ROC_AUC = roc_auc_score(y_test, y_pred1)

# print('ROC AUC : {:.4f}'.format(ROC_AUC))


# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

# Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

# print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

# scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

# print('Cross-validation scores:{}'.format(scores))


# compute Average cross-validation score
# print('Average cross-validation score: {:.4f}'.format(scores.mean()))


from sklearn.model_selection import GridSearchCV
parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]
grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)
grid_search.fit(X_train, y_train)


# examine the best model
# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))
# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))
# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

# calculate GridSearch CV score on test set
print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))