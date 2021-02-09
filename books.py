import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn import datasets
from scipy import sparse
import json


df = pd.read_csv('book_detail.csv', low_memory=True, nrows=5000)
df.drop(df.columns[[-1,-6,-7]], axis=1, inplace=True) 

df.drop(df[df['genre']==''].index, inplace=True)

df = df[df['genre'].notna()] 

genres = []
for i in df['genre']:
    genres.append(list(json.loads(i).values()))
    

df['genre_new'] = genres


df.drop(df.columns[[-2]], axis=1, inplace=True)


df['year'] = df['year'].str[:4]

features = ["author","genre_new","year"] 
for feature in features:
    df[feature] = df[feature].fillna(' ') 

df['genre'] = [' '.join(map(str, l)) for l in df['genre_new']] 

df = df.reset_index(drop=True) 

def combine_features(row):
    '''combines the values of the columns into 1 string'''
    return row['author']+' '+row['genre']+' '+row['year']

df["combined_features"] = df.apply(combine_features,axis=1)

df['index'] = df.index 

cv = CountVectorizer() 
count_matrix = cv.fit_transform(df['combined_features']) #feeding combined strings(movie contents) to CountVectorizer() object
cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    try:
        return df[df.title == title]["index"].values[0]

    except IndexError:
        return f'No book with the name {title}'

def get_five_similar_books(book_user_likes):
    book_index = get_index_from_title(book_user_likes)
    if type(book_index) != np.int64: 
        return f'No book with the name {book_user_likes}'

   
    similar_books = list(enumerate(cosine_sim[book_index])) 

    sorted_similar_books = sorted(similar_books,key=lambda x:x[1],reverse=True)[1:]
    i = 0
    book_array = [] 
    for element in sorted_similar_books:
            book_array.append(get_title_from_index(element[0])) 
            i += 1
            if i >= 5:
                break
    return book_array

if __name__ == '__main__':
    print(get_five_similar_books('abc'))