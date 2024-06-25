from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import requests
import logging
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split


app = Flask(__name__)
api = Api(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
api_url='https://kltn-pescue-production.up.railway.app/api/v1/'

sim_options = {
    'name': 'cosine',
    'user_based': False  # Sử dụng Item-based Collaborative Filtering
}

algo = KNNBasic(sim_options=sim_options)
class Recommend(Resource):
    def get(self, product_id):
        viewed_product, invoice, product = generate_random_data()
        # product_raw = pd.DataFrame(get_products())
        # product = product_raw[['productId', 'categoryName', 'price']]
        # viewed_product_raw = pd.DataFrame(get_view_by_product(product_id))
        # viewed_product = viewed_product_raw[['userId', 'productId']]
        current_product_id = 'P2'
        similar_products = get_similar_products(current_product_id, algo, product, invoice, viewed_product, num_recommendations=5)
        similar_products_json = similar_products.to_json(orient='records')
        return similar_products_json


def generate_random_data(num_users=100, num_products=50, num_invoices=500):
    user_ids = np.arange(1, num_users + 1)
    views = np.random.randint(1, 100, size=num_users)
    viewed_product_data = {
        'user_id': user_ids,
        'view': views
    }
    viewed_product = pd.DataFrame(viewed_product_data)

    product_ids = [f'P{i}' for i in range(1, num_products + 1)]
    category_names = np.random.choice(['Electronics', 'Clothing', 'Accessories', 'Home', 'Toys'], size=num_products)
    prices = np.round(np.random.uniform(5, 500, size=num_products), 2)
    product_data = {
        'product_id': product_ids,
        'categoryName': category_names,
        'price': prices
    }
    product = pd.DataFrame(product_data)

    invoice_user_ids = np.random.choice(user_ids, size=num_invoices)
    invoice_product_ids = np.random.choice(product_ids, size=num_invoices)
    invoice_data = {
        'user_id': invoice_user_ids,
        'product_id': invoice_product_ids
    }
    invoice = pd.DataFrame(invoice_data)

    return viewed_product, invoice, product

# def prepare_data():

#     products["product"] = item_enc.fit_transform(products["productId"])
#     products["category"] = category_enc.fit_transform(products["categoryName"])

#     # Build product profile
#     products["price_category"] = (products["price"] / products["price"].max()) + (
#         products["category"] / products["category"].max()
#     )

#     knn.fit(products[["price_category"]])


# def recommend_products_knn(products_df, products, knn_model, top_n=10):

#     user_data = products_df

#     user_data = user_data.merge(products, on="productId")

#     user_profile = user_data[["price_category"]].mean().values.reshape(1, -1)

#     distances, indices = knn_model.kneighbors(user_profile, n_neighbors=top_n + 1)
#     similar_products = indices.flatten()[1:]

#     recommended_product_ids = products.iloc[similar_products]["productId"].values

#     purchased_products = products_df["productId"].unique()
#     final_recommendations = [
#         prod for prod in recommended_product_ids if prod not in purchased_products
#     ]

#     return final_recommendations[:top_n]


def get_similar_products(product_id, algo, products, invoice, viewed_product, num_recommendations=5):
    users_who_viewed = set(viewed_product['user_id'])
    relevant_invoices = invoice[invoice['user_id'].isin(users_who_viewed)]

    reader = Reader(rating_scale=(1, 1))  
    data = Dataset.load_from_df(relevant_invoices[['user_id', 'product_id', 'user_id']], reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    algo.fit(trainset)
    inner_id = algo.trainset.to_inner_iid(product_id)
    neighbors = algo.get_neighbors(inner_id, k=num_recommendations)

    neighbors = [algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors]

    similar_products = pd.DataFrame(neighbors, columns=['product_id']).merge(products, on='product_id')

    return similar_products[['product_id', 'categoryName', 'price']]

    
def get_products():
    try:
        response = requests.get(f'{api_url}product')
        if response.status_code == 200:
            res = response.json()
            if res['meta']['statusCode'] == '0_2_s':
                return res['data']['productList']
    except Exception as e:
        return {"message": "Error: " + str(e)}

def get_view_by_product(product_id):
    app.logger.info(product_id)
    url = f'{api_url}data/views-audit-log/{product_id}'
    try:
        response = requests.get(url, headers={'client-id' : 'PqescSU7WscLlNRvHK2Ew397vBa0b7dr','client-key': 'opIGrWw2u0WBmZHVIyDRqM6t0P2NKE1c'})
        if response.status_code == 200:
            res = response.json()
            if res['meta']['statusCode'] == '0_2_s':
                return res['data']['views']
    except Exception as e:
        return {"message": "Error: " + str(e)}

api.add_resource(Recommend, "/recommend-product/<product_id>")

if __name__ == "__main__":
    app.run(debug=False)
