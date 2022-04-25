# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:10:19 2022

@author: mesut
"""
# https://www.kaggle.com/midouazerty/restaurant-recommendation-system-using-ml/notebook
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

df = pd.read_csv("zomato.csv")


# Deleting Unnnecessary Columns
df = df.drop(["url", "dish_liked", "phone"], axis = 1)

# Removing the Duplicates (tekar edenleri kaldırmak)
df.duplicated().sum()
df.drop_duplicates(inplace=True)

# Remove the NaN values from the dataset
df.isnull().sum()
df.dropna(how="any", inplace = True)

# Changing the column names
df = df.rename(columns={"approx_cost(for two people)":"cost", "listed_in(type)":"type", "listed_in(city)":"city"})

# Some Transformations
# print(df.dtypes)
df.cost = df.cost.astype(str) # changing the cost to string
df.cost = df.cost.apply(lambda x: x.replace(",", ".")) # using lambda function to replace "," from cost
df.cost = df.cost.astype(float)

df = df.loc[df.rate != "NEW"] # removing "NEW" values lines
df = df.loc[df.rate != "-"].reset_index(drop=True)
remove_slash = lambda x: x.replace("/5", "") if type(x) == np.str else x
df.rate = df.rate.apply(remove_slash).str.strip().astype('float') # removing "/5" from rates

# Adjust the column names
df.name = df.name.apply(lambda x:x.title())
df.online_order.replace(("Yes", "No"),(True, False), inplace = True)
df.book_table.replace(("Yes", "No"),(True, False), inplace = True)

# Computing Mean Rating
restaurants = list(df['name'].unique())
df['Mean Rating'] = 0
for i in range(len(restaurants)):
    df["Mean Rating"][df.name == restaurants[i]] = df.rate[df.name == restaurants[i]].mean()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
df[['Mean Rating']] = scaler.fit_transform(df[['Mean Rating']]).round(2)

# Lower Casing
df["reviews_list"] = df["reviews_list"].str.lower()

# Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", PUNCT_TO_REMOVE))
df["reviews_list"] = df["reviews_list"].apply(lambda text: remove_punctuation(text))

# Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df["reviews_list"] = df["reviews_list"].apply(lambda text: remove_stopwords(text))

# Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
# print(df[['reviews_list', 'cuisines']].sample(5))


# RESTAURANT NAMES:
from sklearn.feature_extraction.text import CountVectorizer
restaurant_names = list(df["name"].unique())
def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range = nu_of_word, stop_words = "english")
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:top_nu_of_words]
df = df.drop(["address", "rest_type", "type", "menu_item", "votes"], axis = 1)

# Randomly sample 60% of your dataframe
df_percent = df.sample(frac=0.5)

df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(name, cosine_similarities = cosine_similarities):
    
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
     
    # Names of the top 30 restaurants
    for each in top30_indexes:
         recommend_restaurant.append(list(df_percent.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))

    return df_new   

print(recommend('Pai Vihar'))
        
        
        
        
        
        
        
        
        
        
        
        