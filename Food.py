import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn import datasets
from scipy import sparse
import json

import pandas as pd 
import numpy as np

df1=pd.read_csv('food.csv')
df1.columns = ['food_id','title','canteen_id','price', 'num_orders', 'category', 'avg_rating', 'num_rating', 'tags']

C= df1['avg_rating'].mean()


m= df1['num_rating'].quantile(0.6)

q_items = df1.copy().loc[df1['num_rating'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['num_rating']
    R = x['avg_rating']
    return (v/(v+m) * R) + (m/(m+v) * C)

q_items['score'] = q_items.apply(weighted_rating, axis=1)

top_rated_items = q_items.sort_values('score', ascending=False)
pop_items= df1.sort_values('num_orders', ascending=False)

def create_soup(x):            
    tags = x['tags'].lower().split(', ')
    tags.extend(x['title'].lower().split())
    tags.extend(x['category'].lower().split())
    return " ".join(sorted(set(tags), key=tags.index))

df1['soup'] = df1.apply(create_soup, axis=1)

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(df1['soup'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices_from_title = pd.Series(df1.index, index=df1['title'])
indices_from_food_id = pd.Series(df1.index, index=df1['food_id'])

def get_recommendations(title="", cosine_sim=cosine_sim, idx=-1):
    if idx == -1 and title != "":
        idx = indices_from_title[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:3]

    food_indices = [i[0] for i in sim_scores]

    return food_indices

# fetch few past orders of a user, based on which personalized recommendations are to be made
def get_latest_user_orders(user_id, orders, num_orders=3):
    counter = num_orders
    order_indices = []
    
    for index, row in orders[['user_id']].iterrows():
        if row.user_id == user_id:
            counter = counter -1
            order_indices.append(index)
        if counter == 0:
            break
            
    return order_indices

# utility function that returns a DataFrame given the food_indices to be recommended
def get_recomms_df(food_indices, df1, columns, comment):
    row = 0
    df = pd.DataFrame(columns=columns)
    
    for i in food_indices:
        df.loc[row] = df1[['title', 'canteen_id', 'price']].loc[i]
        df.loc[row].comment = comment
        row = row+1
    return df

# return food_indices for accomplishing personalized recommendation using Count Vectorizer
def personalised_recomms(orders, df1, user_id, columns, comment="based on your past orders"):
    order_indices = get_latest_user_orders(user_id, orders)
    food_ids = []
    food_indices = []
    recomm_indices = []
    
    for i in order_indices:
        food_ids.append(orders.loc[i].food_id)
    for i in food_ids:
        food_indices.append(indices_from_food_id[i])
    for i in food_indices:
        recomm_indices.extend(get_recommendations(idx=i))
        
    return get_recomms_df(set(recomm_indices), df1, columns, comment)

def get_new_and_specials_recomms(new_and_specials, users, df1, canteen_id, columns, comment="new/today's special item  in your home canteen"):
    food_indices = []
    
    for index, row in new_and_specials[['canteen_id']].iterrows():
        if row.canteen_id == canteen_id:
            food_indices.append(indices_from_food_id[new_and_specials.loc[index].food_id])
            
    return get_recomms_df(set(food_indices), df1, columns, comment)

def get_user_home_canteen(users, user_id):
    for index, row in users[['user_id']].iterrows():
        if row.user_id == user_id:
            return users.loc[index].home_canteen
    return -1


def get_top_rated_items(top_rated_items, df1, columns, comment="top rated items across canteens"):
    food_indices = []
    
    for index, row in top_rated_items.iterrows():
        food_indices.append(indices_from_food_id[top_rated_items.loc[index].food_id])
        
    return get_recomms_df(food_indices, df1, columns, comment)

def get_popular_items(pop_items, df1, columns, comment="most popular items across canteens"):
    food_indices = []
    
    for index, row in pop_items.iterrows():
        food_indices.append(indices_from_food_id[pop_items.loc[index].food_id])
        
    return get_recomms_df(food_indices, df1, columns, comment)

orders = pd.read_csv('orders.csv')
new_and_specials = pd.read_csv('new_and_specials.csv')
users = pd.read_csv('users.csv')

columns = ['title', 'canteen_id', 'price', 'comment']
current_user = 2
current_canteen = get_user_home_canteen(users, current_user)

def get_title_from_index(index):
    return df1[df1.index == index]["title"].values[0]

def get_index_from_title(title):
    try:
        return df1[df1.title == title]["index"].values[0]

    except IndexError:
        return f'No food with the name {title}'

def get_five_similar_food(food_user_likes):
    food_index = get_index_from_title(food_user_likes)
    if type(food_index) != np.int64: 
        return f'No food with the name {food_user_likes}'

   
    similar_foods = list(enumerate(cosine_sim[food_index])) 

    sorted_similar_foods = sorted(similar_foods,key=lambda x:x[1],reverse=True)[1:]
    i = 0
    book_array = [] 
    for element in sorted_similar_foods:
            book_array.append(get_title_from_index(element[0])) 
            i += 1
            if i >= 5:
                break
    return book_array

if __name__ == '__main__':
    print(get_five_similar_food('Chicken Tikka'))
    