# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:31:42 2022

@author: mesut
"""
# https://www.kaggle.com/midouazerty/book-recommendation-system-with-machine-learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

books = pd.read_csv("BX_Books.csv", sep = ";")
ratings = pd.read_csv("BX-Book-Ratings.csv", sep = ";")
users = pd.read_csv("BX-Users.csv", sep = ";")

# plt.rc("font", size =15)
# ratings["Book-Rating"].value_counts(sort=False).plot(kind="bar")
# plt.title('Rating Distribution\n')
# plt.xlabel('Rating')
# plt.ylabel('Count')
# plt.show()

# users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
# plt.title('Age Distribution\n')
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.show()

counts1 = ratings['User-ID'].value_counts()
ratings = ratings[ratings['User-ID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['Book-Rating'].value_counts()
ratings = ratings[ratings['Book-Rating'].isin(counts[counts >= 100].index)]

combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['Year-Of-Publication', 'Publisher', 'Book-Author', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
combine_book_rating = combine_book_rating.drop(columns, axis=1)


combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['Book-Title'])
book_ratingCount = (combine_book_rating.
     groupby(by = ['Book-Title'])['Book-Rating'].
     count().
     reset_index().
     rename(columns = {'Book-Rating': 'totalRatingCount'})
     [['Book-Title', 'totalRatingCount']] )

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'Book-Title', right_on = 'Book-Title', how = 'left')
   
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# print(book_ratingCount['totalRatingCount'].describe())

# print(book_ratingCount['totalRatingCount'].quantile(np.arange(0.9, 1, .01)))

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')


combined = rating_popular_book.merge(users, left_on = 'User-ID', right_on = 'User-ID', how = 'left')
us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)

from scipy.sparse import csr_matrix
us_canada_user_rating = us_canada_user_rating.drop_duplicates(['User-ID', 'Book-Title'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)

query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
# print(query_index)
# print(us_canada_user_rating_pivot.iloc[query_index,:].values.reshape(1,-1))
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
us_canada_user_rating_pivot.index[query_index]

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))





