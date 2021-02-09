import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets

df = pd.read_csv("wayfair_uniq.csv")
df_new=df.iloc[:,[3,4,5,7,9,11,13,15,17,18,19]]

df_new['index'] = df_new.index


df_new['original_price'] = df_new.original_price.str.replace('$', '')
df_new['original_price'] = df_new.original_price.str.replace(',', '').astype(float)

new_price = df_new["price"].str.split(";\s\$", n = 1, expand = True)
df_new["new_price"] = new_price[1]
df_new.drop(columns =["price"], inplace = True)

price_range = df_new[df_new["new_price"].str.contains("-",regex=True)]

price_range['new_price'] = (price_range['new_price'].str.split("\s-", n = 1, expand = True))[0]

df_new.loc[price_range.index] = price_range

df_new['new_price'] = df_new.new_price.str.replace(',', '').astype(float)

bins = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,35000]
labels = ["<$500","$500-$1000","$1000-$1500","$1500-$2000","$2000-$2500","$2500-$3000","$3000-$3500","$3500-$4000","$4000-$4500",">$4500"]
df_new['price_bracket'] = pd.cut(df_new['new_price'], bins, labels=labels)


features = ["product_data1","product_data2","product_descrip","manufacturer"]

for feature in features:
    df_new[feature] = df_new[feature].fillna('')

def combine_features(row):
    try:
        return row['product_data1'] +" "+row['product_data2']+" "+row["product_descrip"]+" "+row["price_bracket"]+" "+row["manufacturer"]
    except:
        print("Error:", row)

df_new["combined_features"] = df_new.apply(combine_features,axis=1)

stop_words = stopwords.words('english')

df_new['combined_features'] = df_new['combined_features'].str.lower().str.split()

df_new["features"]=df_new["combined_features"].apply(lambda x: [word for word in x if word not in stop_words])

df_new["features"]=df_new["features"].apply(lambda x: " ".join(x))

cv = CountVectorizer()

count_matrix = cv.fit_transform(df_new["features"])

cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df_new[df_new.index == index]["product_name"].values[0]

def get_home(index):
    return df_new[df_new.index == index]["product_name_link"].values[0]

def get_index_from_title(title):
    return df_new[df_new.product_name == title]["index"].values[0]

def get_five_similar_products(product_user_likes):
    product_index = get_index_from_title(product_user_likes)
    if type(product_index) != np.int64:
        return f'No movie with the name {product_user_likes}'

    similar_product = list(enumerate(cosine_sim[product_index])) 

    sorted_similar_product = sorted(similar_product,key=lambda x:x[1],reverse=True)[1:]

    i = 0
    product_array = [] 
    for element in sorted_similar_product:
            product_array.append(get_title_from_index(element[0])) 
            i += 1
            if i >= 5:
                break

    return product_array

if __name__ == '__main__':
    print(get_five_similar_products('Bjorn Chesterfield Settee'))

