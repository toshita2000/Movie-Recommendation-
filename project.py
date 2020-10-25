# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:36:54 2020

@author: Toshita Sharma
"""

import pandas as pd
import numpy as np

movies_df = pd.read_csv('C:/Users/Toshita Sharma/Desktop/SP Innovative assignment/movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
rating_df=pd.read_csv('C:/Users/Toshita Sharma/Desktop/SP Innovative assignment/ratings.csv',usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

print(movies_df.head())

print(rating_df.head())


df = pd.merge(rating_df,movies_df,on='movieId')
print(df.head())

combine_movie_rating = df.dropna(axis = 0, subset = ['title'])
movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )
movie_ratingCount.head()
rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
print(rating_with_totalRatingCount.head())


pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(movie_ratingCount['totalRatingCount'].describe())



popularity_threshold = 50
rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
print(rating_popular_movie.head())


print(rating_popular_movie.shape)

movie_features_df=rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)
movie_features_df.head()

from scipy.sparse import csr_matrix

movie_features_df_matrix = csr_matrix(movie_features_df.values)

from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
print(model_knn.fit(movie_features_df_matrix))

print(movie_features_df.shape)

query_index = np.random.choice(movie_features_df.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)


print(movie_features_df.head())

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))


import streamlit as st
st.title("Movie Recommendation")
st.sidebar.title("Movie Recommendation")
st.sidebar.image("image2.jpg", use_column_width=True)
st.sidebar.subheader("Scientific Programming -  2CSOE76✨")
st.sidebar.subheader("Made by: - 18BEC115 & 18BEC118")
    
st.sidebar.subheader("This model works on KNN -    K nearest neighbour algorithm to recommend movies based on ratings. This is called collaborative filtering.")
st.image("image1.jpg", use_column_width=True)
st.header("")
for i in range(0, len(distances.flatten())):
    if i == 0:
        st.write('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        st.write('{0}: {1}'.format(i, movie_features_df.index[indices.flatten()[i]]))



