# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 23:28:50 2022

@author: mesut
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA #**************
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


df= pd.read_csv("Housing.csv")
print(df.shape)
# print(df.info())
describe = df.describe()
# print(df.describe())

# Checking Null variables (null değişken yoklaması)
Null = df.isnull().sum()*100/df.shape[0]
    # There are no NULL values in the dataset, hence it is clean.
# dataset içinde Null değer yok. 

# Outlier Analysis

# fig, axs = plt.subplots(2,3, figsize = (10,5))
# plt1 = sns.boxplot(df.price, ax = axs[0,0])
# plt2 = sns.boxplot(df.area, ax = axs[0,1])
# plt3 = sns.boxplot(df.bedrooms, ax = axs[0,2])
# plt4 = sns.boxplot(df.bathrooms, ax = axs[1,0])
# plt5 = sns.boxplot(df.stories, ax = axs[1,1])
# plt6 = sns.boxplot(df.parking, ax = axs[1,2])
# plt.tight_layout()
 
# Outlier Treatment
# Price and area have considerable outliers.
# We can drop the outliers as we have sufficient data.

# outlier treatment for price
plt.boxplot(df.price)
Q11 = df.price.quantile(0.25)
Q33 = df.price.quantile(0.75)
IQR1 = Q33 - Q11
df = df[(df.price >= Q11 - 1.5*IQR1) & (df.price <= Q33 + 1.5*IQR1)]
print(df.shape)

# outlier treatment for area
plt.boxplot(df.area)
Q1 = df.area.quantile(0.25)
Q3 = df.area.quantile(0.75)
IQR = Q3 - Q1
df = df[(df.area >= Q1 - 1.5*IQR) & (df.area <= Q3 + 1.5*IQR)]
print(df.shape)

# fig, axs = plt.subplots(2,3, figsize = (10,5))
# plt1 = sns.boxplot(df.price, ax = axs[0,0])
# plt2 = sns.boxplot(df.area, ax = axs[0,1])
# plt3 = sns.boxplot(df.bedrooms, ax = axs[0,2])
# plt4 = sns.boxplot(df.bathrooms, ax = axs[1,0])
# plt5 = sns.boxplot(df.stories, ax = axs[1,1])
# plt6 = sns.boxplot(df.parking, ax = axs[1,2])
# plt.tight_layout()

# sns.pairplot(df)
# plt.show()

# plt.figure(figsize=(20, 12))
# plt.subplot(2,3,1)
# sns.boxplot(x = 'mainroad', y = 'price', data = df)
# plt.subplot(2,3,2)
# sns.boxplot(x = 'guestroom', y = 'price', data = df)
# plt.subplot(2,3,3)
# sns.boxplot(x = 'basement', y = 'price', data = df)
# plt.subplot(2,3,4)
# sns.boxplot(x = 'hotwaterheating', y = 'price', data = df)
# plt.subplot(2,3,5)
# sns.boxplot(x = 'airconditioning', y = 'price', data = df)
# plt.subplot(2,3,6)
# sns.boxplot(x = 'furnishingstatus', y = 'price', data = df)
# plt.show()

# We can also visualise some of these categorical features parallely by using the hue argument.
# Below is the plot for furnishingstatus with airconditioning as the hue.

# plt.figure(figsize = (10, 5))
# sns.boxplot(x = 'furnishingstatus', y = 'price', hue = 'airconditioning', data = df)
# plt.show()

# Data Preparation

# You can see that your dataset has many columns with values as 'Yes' or 'No'.

# But in order to fit a regression line, we would need numerical values and not string.
# Hence, we need to convert them to 1s and 0s, where 1 is a 'Yes' and 0 is a 'No'.

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
def binary_map(x):
    return x.map({'yes': 1, "no": 0})
df[varlist] = df[varlist].apply(binary_map)

# Dummy Variables
# The variable furnishingstatus has three levels. We need to convert these levels into integer as well.
# For this, we will use something called dummy variables.

# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'

status = pd.get_dummies(df['furnishingstatus'])

# Now, you don't need three columns. You can drop the furnished column,
# as the type of furnishing can be identified with just the last two columns where
status = pd.get_dummies(df['furnishingstatus'], drop_first = True)
df = pd.concat([df, status], axis = 1)

# Drop 'furnishingstatus' as we have created the dummies for it
df.drop(['furnishingstatus'], axis = 1, inplace = True)

np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)

# As you know, there are two common ways of rescaling:
    # Min-Max scaling
    # Standardisation (mean-0, sigma-1)
# This time, we will use MinMax scaling.
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

# Let's check the correlation coefficients to see which variables are highly correlated
# plt.figure(figsize = (16, 10))
# sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
# plt.show()

# As you might have noticed, area seems to the correlated to price the most. 
# Let's see a pairplot for area vs price.
y_train = df_train.pop('price')
X_train = df_train

# This time, we will be using the LinearRegression function from SciKit Learn for its compatibility with RFE (which is a utility from sklearn)

#  RFE: Recursive feature elimination
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm, 6)             # running RFE
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]






     










