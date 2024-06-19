import requests
import pandas as pd
from pandas import json_normalize

def get_data(API, header):
    """Function to get data from the given API and return a DataFrame"""
    r = requests.request('GET', API, headers=header)
    json_data = r.json()
    df = json_normalize(json_data)
    return df

def split_user_data(df_user):
    """Function to split user data into two DataFrames based on count_num_order"""
    data_user_for_model1 = df_user[df_user['count_num_order'] <= 5].drop(columns=['count_num_order'])
    data_user_for_model2 = df_user[df_user['count_num_order'] > 5].drop(columns=['count_num_order'])
    return data_user_for_model1, data_user_for_model2

def main():
    # URLs for the product and user data
    header = {"Authorization": "Sudah izin pada Wildan dan Yoga"}

    product_url = 'http://34.101.249.106/product'
    user_url = 'http://34.101.249.106/pref'
    interaction_url = 'http://34.101.249.106/order'

    # Retrieve data from the given URLs
    df_product = get_data(product_url, header)
    df_user = get_data(user_url, header)
    df_interaction = get_data(interaction_url, header)

    # # Split user data into two models based on count_num_order
    data_user_for_model1, data_user_for_model2 = split_user_data(df_user)

    # Save data to Excel files
    with pd.ExcelWriter('all_data.xlsx') as writer:
        df_user.to_excel(writer, sheet_name='user', index=False)
        df_product.to_excel(writer, sheet_name='product', index=False)

    with pd.ExcelWriter('data_for_model1.xlsx') as writer:
        data_user_for_model1.to_excel(writer, sheet_name='user', index=False)
        df_product.to_excel(writer, sheet_name='product', index=False)

    with pd.ExcelWriter('data_for_model2.xlsx') as writer:
        data_user_for_model2.to_excel(writer, sheet_name='user', index=False)
        df_product.to_excel(writer, sheet_name='product', index=False)
        df_interaction.to_excel(writer, sheet_name='interaction', index=False)

# Run the main function
if __name__ == "__main__":
    main()
