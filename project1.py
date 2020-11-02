# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 23:10:00 2020

@author: Toshita Sharma
"""

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns
movies = pd.read_csv('C:/Users/Toshita Sharma/Desktop/SP Innovative assignment/movies.csv')
ratings=pd.read_csv('C:/Users/Toshita Sharma/Desktop/SP Innovative assignment/ratings.csv')

ratings.head()
movies.head()

final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()


final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()


#final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]

sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
print(sparsity)

csr_sample = csr_matrix(sample)
print(csr_sample)


csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)


knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),\
                               key=lambda x: x[1])[:0:-1]
        
        recommend_frame = []
        
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    
    else:
        
        return "No movies found. Please check your input"




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
movies1 = pd.read_csv('C:/Users/Toshita Sharma/Desktop/SP Innovative assignment/movies.csv')
ratings1=pd.read_csv('C:/Users/Toshita Sharma/Desktop/SP Innovative assignment/ratings.csv')
genres=[]
for genre in movies1.genres:
    
    x=genre.split('|')
    for i in x:
         if i not in genres:
            genres.append(str(i))
genres=str(genres)    
movie_title=[]
for title in movies1.title:
    movie_title.append(title[0:-7])
movie_title=str(movie_title)    
df=pd.merge(ratings1,movies1, how='left',on='movieId')
df.head()
df1=df.groupby(['title'])[['rating']].sum()
high_rated=df1.nlargest(20,'rating')
high_rated.head()
plt.figure(figsize=(30,10))
plt.title('Top 20 movies with highest rating',fontsize=40)
colors=['red','yellow','orange','green','magenta','cyan','blue','lightgreen','skyblue','purple']
plt.ylabel('ratings',fontsize=30)
plt.xticks(fontsize=25,rotation=90)
plt.xlabel('movies title',fontsize=30)
plt.yticks(fontsize=25)
plt.bar(high_rated.index,high_rated['rating'],linewidth=3,edgecolor='red',color=colors)

df2=df.groupby('title')[['rating']].count()
rating_count_20=df2.nlargest(20,'rating')
rating_count_20.head()

plt.figure(figsize=(30,10))
plt.title('Top 20 movies with highest number of ratings',fontsize=30)
plt.xticks(fontsize=25,rotation=90)
plt.yticks(fontsize=25)
plt.xlabel('movies title',fontsize=30)
plt.ylabel('ratings',fontsize=30)

plt.bar(rating_count_20.index,rating_count_20.rating,color='red')

cv=TfidfVectorizer()
tfidf_matrix=cv.fit_transform(movies['genres'])
movie_user = df.pivot_table(index='userId',columns='title',values='rating')
movie_user.head()


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices=pd.Series(movies1.index,index=movies1['title'])
titles=movies['title']
def recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]





import streamlit as st
st.title("Movie Recommendation")
st.sidebar.title("Movie Recommendation")
st.sidebar.image("image2.jpg", use_column_width=True)
st.sidebar.subheader("Scientific Programming -  2CSOE76✨")
st.sidebar.subheader("Made by: - 18BEC115 & 18BEC118")
    
#st.sidebar.subheader("This model works on KNN -    K nearest neighbour algorithm to recommend movies based on ratings. This is called collaborative filtering.")
st.image("image1.jpg", use_column_width=True)
st.header("")
classifier= st.sidebar.selectbox("classifier",("KNN","TFID"))
if classifier == 'KNN':
    input_msg = st.text_input("")
    st.subheader("Press enter to check the prediction...")
    st.write(get_movie_recommendation(input_msg))
    
if classifier == 'TFID':
    input_msg1=st.text_input("")
    st.subheader("Press enter to check the prediction...")
    st.write(recommendations(input_msg1))




