from flask import Flask, jsonify
from pymongo import MongoClient
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import json
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

@app.route('/')
def home():
    return "Machine Learning Server is running..."

@app.route('/forecast/<product_name>', methods=['GET'])
def forecast(product_name):
    # Establish database connection
    client = MongoClient("mongodb+srv://kalindukadikari:HTsxXUhM6syHvJ69@appofsos.eahoi5d.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true")
    db = client.test
    collection = db.orders

    # Load data from MongoDB
    data = list(collection.find())

    # Update '_id' in 'items' to 'item_id'
    for document in data:
        for item in document['items']:
            item['item_id'] = item.pop('_id')

    # Convert MongoDB data to DataFrame
    df = pd.json_normalize(data, record_path=['items'], meta=['_id', 'createdAt'], meta_prefix='order_')

    # Convert 'name' and 'order__id' to string
    df['name'] = df['name'].astype(str)
    df['order__id'] = df['order__id'].astype(str)

    # Convert 'quantity' to int and 'createdAt' to datetime
    df['quantity'] = df['quantity'].astype(int)
    df['order_createdAt'] = pd.to_datetime(df['order_createdAt'])

    # Group by 'name' and 'order_createdAt' and sum 'quantity'
    df_grouped = df.groupby(['name', df['order_createdAt'].dt.date])['quantity'].sum().reset_index()

    # Forecast for the provided product
    df_product = df_grouped[df_grouped['name'] == product_name].set_index('order_createdAt')

    # Fit the ARIMA model
    model = ARIMA(df_product['quantity'], order=(5,1,0))
    model_fit = model.fit()

    # Forecast the next 30 days
    forecast = model_fit.get_forecast(steps=30)
    predicted = forecast.predicted_mean 

    # Convert forecast to a list and return as JSON
    result = [{"day": (pd.Timestamp.today() + pd.DateOffset(days=day)).strftime('%Y-%m-%d'), 
               "forecast": round(forecast_value)} for day, forecast_value in enumerate(predicted)]
    return jsonify(result), 200



@app.route('/best_selling_products', methods=['GET'])
def best_selling_products():
    # Establish database connection
    client = MongoClient("mongodb+srv://kalindukadikari:HTsxXUhM6syHvJ69@appofsos.eahoi5d.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true")
    db = client.test
    collection = db.orders

    # Load data from MongoDB
    data = list(collection.find())

    # Update '_id' in 'items' to 'item_id'
    for document in data:
        for item in document['items']:
            item['item_id'] = item.pop('_id')

    # Convert MongoDB data to DataFrame
    df = pd.json_normalize(data, record_path=['items'], meta=['_id'], meta_prefix='order_')

    # Convert 'name' and 'order__id' to string
    df['name'] = df['name'].astype(str)
    df['order__id'] = df['order__id'].astype(str)

    # Convert 'quantity' to int
    df['quantity'] = df['quantity'].astype(int)

    # Calculate the total quantity sold for each product
    product_sales = df.groupby('name')['quantity'].sum()

    # Sort products by total quantity sold in descending order
    best_selling_products = product_sales.sort_values(ascending=False)

    # Convert Series to a list of dictionaries
    result = [{"name": name, "quantity": quantity} for name, quantity in best_selling_products.items()]

    return jsonify(result), 200

@app.route('/best_selling', methods=['GET'])
def best_selling():
    # Establish database connection
    client = MongoClient("mongodb+srv://kalindukadikari:HTsxXUhM6syHvJ69@appofsos.eahoi5d.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true")
    db = client.test
    collection = db.orders

    # Load data from MongoDB
    data = list(collection.find())

    # Update '_id' in 'items' to 'item_id'
    for document in data:
        for item in document['items']:
            item['item_id'] = item.pop('_id')

    # Convert MongoDB data to DataFrame
    df = pd.json_normalize(data, record_path=['items'], meta=['_id'], meta_prefix='order_')

    # Convert 'name' and 'order__id' to string
    df['name'] = df['name'].astype(str)
    df['order__id'] = df['order__id'].astype(str)

    # Convert 'quantity' to int
    df['quantity'] = df['quantity'].astype(int)

    # One-hot encoding for Apriori Algorithm
    df_onehot = df.pivot_table(index='order__id', columns='name', values='quantity', aggfunc='sum').fillna(0)

    # Convert quantity to boolean for Apriori Algorithm
    df_onehot = df_onehot.applymap(lambda quantity: 1 if int(quantity) >= 1 else 0)
    df_onehot = df_onehot.astype(bool)

    # Run the apriori algorithm and generate the frequent itemsets
    frequent_itemsets = apriori(df_onehot, min_support=0.01, use_colnames=True) # Adjust the min_support value as needed

    # Generate the association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Filter the DataFrame for better readability
    rules = rules.filter(items=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

    # Convert DataFrame to JSON
    result = rules.to_json(orient="records")
    parsed = json.loads(result)

    return jsonify(parsed), 200


if __name__ == '__main__':
    app.run(debug=True)
