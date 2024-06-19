import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

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


# Function to merge user and product data
def merge_user_product_data(df_user, df_product):
    df_user['key'] = 0
    df_product['key'] = 0
    merged_data = pd.merge(df_user, df_product, on='key').drop('key', axis=1)
    return merged_data

def add_similarity_column(merged_data):
    user_cols = merged_data.columns[1:21]
    product_cols = merged_data.columns[22:]

    merged_data['similarity_value'] = 0

    for i in range(len(merged_data)):
        similarity_dict = {}
        for user_col, product_col in zip(user_cols, product_cols):
            if merged_data[user_col][i] == 1 and merged_data[product_col][i] == 1:
                similarity_value = similarity_dict.get((user_col, product_col), 0)
                similarity_dict[(user_col, product_col)] = similarity_value + 1

        merged_data.at[i, 'similarity_value'] = sum(similarity_dict.values())

    return merged_data

def create_and_train_model(merged_data):
    features = merged_data.drop(columns=['user_id', 'product_id'])
    X = features.drop(columns=['similarity_value']).values.astype(float)
    y = features['similarity_value'].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=7, batch_size=32, validation_data=(X_test, y_test))
    model.save('model1.h5')
    return model

# Main function to run the recommendation system
def main():
    # Paths
    file_path = '/home/c369d4ky1284/ml/data_for_model1.xlsx'

    # Load and preprocess data
    df_product, df_user = load_data(file_path)
    df_product = preprocess_product_data(df_product)

    feature_columns = df_product.columns.tolist()
    feature_columns = [x for x in feature_columns if x != 'product_id']
    df_user = preprocess_user_data(df_user, feature_columns)

    # Merge user and product data
    merged_data = merge_user_product_data(df_user, df_product)

    # Add similarity column
    merged_data = add_similarity_column(merged_data)

    # Create and train model
    model = create_and_train_model(merged_data)

if __name__ == "__main__":
    main()
