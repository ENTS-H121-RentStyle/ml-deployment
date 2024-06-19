import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import requests

# Function to load data
def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    df_product = pd.read_excel(xls, sheet_name='product')
    df_user = pd.read_excel(xls, sheet_name='user')
    return df_product, df_user

# Function to preprocess product data
def preprocess_product_data(df_product):
    # Convert text to lowercase and split categorical data into lists
    df_product['category'] = df_product['category'].apply(lambda x: x.lower().split(', '))
    df_product['color'] = df_product['color'].apply(lambda x: x.lower().split(', '))
    df_product['size'] = df_product['size'].apply(lambda x: x.lower().split(', '))

    # Initialize and apply MultiLabelBinarizer
    mlb_product = MultiLabelBinarizer()

    # Transform 'category'
    category_encoded = mlb_product.fit_transform(df_product['category'])
    category_df = pd.DataFrame(category_encoded, columns=mlb_product.classes_)

    # Transform 'color'
    color_encoded = mlb_product.fit_transform(df_product['color'])
    color_df = pd.DataFrame(color_encoded, columns=mlb_product.classes_)

    # Transform 'size'
    size_encoded = mlb_product.fit_transform(df_product['size'])
    size_df = pd.DataFrame(size_encoded, columns=mlb_product.classes_)

    # Join the encoded dataframes back to the original dataframe
    df_product = df_product.join(category_df).join(color_df).join(size_df)

    # Drop unnecessary columns
    df_product.drop(columns=['category', 'color', 'size', 'product_name', 'rent_price', 'count_num_rating', 'avg_rating', 'count_num_order'], inplace=True)

    return df_product

# Function to preprocess user data
def preprocess_user_data(df_user, feature_columns):
    # Convert non-string entries to an empty string
    df_user['category_preference'] = df_user['category_preference'].apply(lambda x: x.lower().split(', ') if isinstance(x, str) else [])
    df_user['color_preference'] = df_user['color_preference'].apply(lambda x: x.lower().split(', ') if isinstance(x, str) else [])
    df_user['size_preference'] = df_user['size_preference'].apply(lambda x: x.lower().split(', ') if isinstance(x, str) else [])

    # Initialize and apply MultiLabelBinarizer
    mlb_user = MultiLabelBinarizer()

    # Transform 'category_preference'
    category_preference_encoded = mlb_user.fit_transform(df_user['category_preference'])
    category_preference_df = pd.DataFrame(category_preference_encoded, columns=mlb_user.classes_)

    # Transform 'color_preference'
    color_preference_encoded = mlb_user.fit_transform(df_user['color_preference'])
    color_preference_df = pd.DataFrame(color_preference_encoded, columns=mlb_user.classes_)

    # Transform 'size_preference'
    size_preference_encoded = mlb_user.fit_transform(df_user['size_preference'])
    size_preference_df = pd.DataFrame(size_preference_encoded, columns=mlb_user.classes_)

    # Join the encoded dataframes back to the original dataframe
    df_user = df_user.join(category_preference_df).join(color_preference_df).join(size_preference_df)

    # Drop unnecessary columns
    df_user.drop(columns=['category_preference', 'color_preference', 'size_preference', 'count_num_rating_user', 'avg_rating_user'], inplace=True)

    # Ensure all feature columns in df_user match those in df_product
    for column in feature_columns:
        if column not in df_user.columns:
            df_user[column] = 0

    # Reorder columns in df_user to match df_product
    df_user = df_user[['user_id'] + feature_columns]

    return df_user

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def post_data(API, header, data):
    response = requests.post(API, headers=header, json=data)
    return response

def recommend_products_n(user_id, df_user, df_product, model, API, header, n=150):
    user_row = df_user[df_user['user_id'] == user_id].drop(columns=['user_id'])
    product_features = df_product.drop(columns=['product_id'])

    user_row['key'] = 0
    product_features['key'] = 0
    combined_features = pd.merge(user_row, product_features, on='key').drop(columns='key').values

    predictions = model.predict(combined_features)
    max_indices = [max(enumerate(prediction), key=lambda x: x[1])[0] for prediction in predictions]

    product_ids = df_product['product_id']
    recommended_products = pd.DataFrame({'product_id': product_ids, 'similarity_value': max_indices})

    recommended_products = recommended_products.sort_values(by='similarity_value', ascending=False)
    recommended_list = recommended_products.head(n)['product_id'].values.tolist()

    result = {
        'user_id': user_id,
        'recommendation': recommended_list,
        'model_type': 'model1'}

    # Post the result using post_data function
    response = post_data(API, header, result)
    return response.json()

def main():
    # Paths
    file_path = '/home/c369d4ky1284/ml/data_for_model1.xlsx'

    # Load and preprocess data
    df_product, df_user = load_data(file_path)
    df_product = preprocess_product_data(df_product)

    # Extract feature columns from df_product
    feature_columns = df_product.columns.tolist()
    feature_columns = [x for x in feature_columns if x != 'product_id']

    df_user = preprocess_user_data(df_user, feature_columns)

    # Load the model
    model = load_model('/home/c369d4ky1284/ml/model1.h5')

    header = {"Authorization": "Sudah izin pada Wildan dan Yoga"}
    API = 'http://34.101.249.106/result/model1'

    # Iterate through each user and get recommendations
    for user_id in df_user['user_id']:
        response = recommend_products_n(user_id, df_user, df_product, model, API, header, 30)
        print(f'Recommendation response for user {user_id}:', response)

if __name__ == "__main__":
    main()
