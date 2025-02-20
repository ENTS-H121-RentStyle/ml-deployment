# -*- coding: utf-8 -*-
"""model2_inference_script.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vwUeccEO6IBeUKF7O9UyOk4YfdZjb0yo
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Function to preprocess data
def preprocess_data(file_path):
    xls = pd.ExcelFile(file_path)
    products = pd.read_excel(xls, sheet_name='product')

    # Handle missing values
    products['product_name'] = products['product_name'].fillna('')
    products['category'] = products['category'].fillna('')
    products['color'] = products['color'].fillna('')
    products['size'] = products['size'].fillna('')
    products['rent_price'] = products['rent_price'].fillna(0).astype(float)
    products['count_num_rating'] = products['count_num_rating'].fillna(0).astype(int)
    products['avg_rating'] = products['avg_rating'].fillna(0).astype(float)

    # Combine text and numerical features into a single feature
    products['features'] = (products['product_name'] + ' ' +
                            products['category'] + ' ' +
                            products['color'] + ' ' +
                            products['size'] + ' ' +
                            products['rent_price'].astype(str) + ' ' +
                            products['count_num_rating'].astype(str) + ' ' +
                            products['avg_rating'].astype(str))

    return products

# Function to preprocess data, vectorize, scale, combine, and calculate similarity
def vectorize_scale_combine_calculate(products):
    # Vectorize text features
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(products['features'])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(products[['rent_price', 'count_num_rating', 'avg_rating']])

    # Combine TF-IDF matrix and scaled numerical features
    features_matrix = np.hstack((tfidf_matrix.toarray(), numerical_features))

    # Calculate similarity scores (cosine similarity)
    similarity_matrix = cosine_similarity(features_matrix, features_matrix)

    return features_matrix, similarity_matrix

# Function to load a pre-trained model from .h5 file
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to get top N recommendations for a given user ID
def get_top_n_recommendations_based_user_interaction(user_id, similarity_matrix, interactions_data, products_data, n=5):
    # Filter interactions for the given user
    user_interactions = interactions_data[interactions_data['user_id'] == user_id]

    # Initialize an empty dictionary to store aggregated similarity scores
    agg_scores = {}

    # Iterate through each product the user has interacted with
    for index, row in user_interactions.iterrows():
        product_id = row['product_id']
        # Get the similarity scores for the product
        idx = products_data.index[products_data['product_id'] == product_id].tolist()[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))

        # Add the similarity scores to the aggregated dictionary
        for i, score in sim_scores:
            if i not in agg_scores:
                agg_scores[i] = score
            else:
                agg_scores[i] += score

    # Sort the aggregated similarity scores
    sorted_scores = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)

    # Get the top N recommended products
    top_n_indices = [i for i, _ in sorted_scores[:n]]

    return products_data['product_id'].iloc[top_n_indices]

def post_data(API, header, data):
    response = requests.post(API, headers=header, json=data)
    return response

# Function to get recommended products as a DataFrame
def get_recommended_products_dataframe(user_id, product_ids, products_data, API, header):
    # Filter products DataFrame based on recommended product IDs
    recommended_products = products_data[products_data['product_id'].isin(product_ids)]
    recommended_list = recommended_products['product_id'].values.tolist()

    result = {
        'user_id': user_id,
        'recommendation': recommended_list,
        'model_type': 'model2'}

    # Post the result using post_data function
    response = post_data(API, header, result)
    return response.json()

# Function to execute the entire process
def main(file_path, model_path, API, header):
    # Preprocess data
    products = preprocess_data(file_path)

    # Vectorize, scale, combine, and calculate similarity
    features_matrix, similarity_matrix = vectorize_scale_combine_calculate(products)

    # Load pre-trained model
    trained_model = load_model(model_path)

    # Assuming interactions data is also loaded
    interactions = pd.read_excel(file_path, sheet_name='interaction')

    # Iterate through each user and get recommendations
    unique_users = interactions['user_id'].unique()
    for user_id in unique_users:
        top_20_recommendations_for_user = get_top_n_recommendations_based_user_interaction(user_id, similarity_matrix, interactions, products, n=20)
        response = get_recommended_products_dataframe(user_id, top_20_recommendations_for_user, products, API, header)
        print(f"Recommendation response for user {user_id}:", response)

if __name__ == "__main__":
    file_path = '/home/c369d4ky1284/ml/data_for_model2.xlsx'  # feel free to change it
    model_path = '/home/c369d4ky1284/ml/model2.h5'  # feel free to change it
    header = {"Authorization": "Sudah izin pada Wildan dan Yoga"}
    API = 'http://34.101.249.106/result/model2'
    main(file_path, model_path, API, header)
