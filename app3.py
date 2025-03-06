import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model, scaler, dataset, and feature list
model = joblib.load('sales_forecast_model_3.pkl')
scaler = joblib.load('scaler_forecast_3.pkl')
feature_list = joblib.load('feature_list_2.pkl')
df = pd.read_csv("sales_forecast_with_performance_adjustments_2.csv")

# Streamlit UI
st.title("📈 Amazon Product Sales Predictor")
st.sidebar.header("🛍️ Enter Product Details")

# Product Title Search
product_title = st.sidebar.text_input("🔍 Enter Product Title (Start Typing...)")

# Filter product titles that match the input text
matching_titles = df[df['product_title'].str.contains(product_title, case=False, na=False)]['product_title'].unique()

if product_title and len(matching_titles) > 0:
    selected_title = st.sidebar.selectbox("📜 Select Matching Title", matching_titles)
    product_data = df[df['product_title'] == selected_title]
    asin = product_data['asin'].values[0]
else:
    selected_title = None
    asin = st.sidebar.text_input("🔍 Enter ASIN (Optional for Autofill)")

# Autofill details based on ASIN or selected title
if asin:
    product_data = df[df['asin'] == asin]
    if not product_data.empty:
        st.sidebar.success("✅ Product Found! Autofilling Data.")
        previous_month_sales = int(product_data['sales_volume_previous_month'].values[0])
        two_months_ago_sales = int(product_data['sales_volume_next_month'].values[0])
        product_price = float(product_data['product_price'].values[0])
        product_star_rating = float(product_data['product_star_rating'].values[0])
        product_num_ratings = int(product_data['product_num_ratings'].values[0])
        product_num_offers = int(product_data['product_num_offers'].values[0])
        is_best_seller = int(product_data['is_best_seller'].values[0])
        is_prime = int(product_data['is_prime'].values[0])
    else:
        st.sidebar.warning("⚠️ ASIN not found! Enter details manually.")
        previous_month_sales, two_months_ago_sales, product_price, product_star_rating = 0, 0, 0.0, 0.0
        product_num_ratings, product_num_offers, is_best_seller, is_prime = 0, 0, 0, 0
else:
    previous_month_sales, two_months_ago_sales, product_price, product_star_rating = 0, 0, 0.0, 0.0
    product_num_ratings, product_num_offers, is_best_seller, is_prime = 0, 0, 0, 0

# User Input Fields (Editable)
previous_month_sales = st.sidebar.number_input("📊 Previous Month Sales", min_value=0, value=previous_month_sales)
two_months_ago_sales = st.sidebar.number_input("📉 Two Months Ago Sales", min_value=0, value=two_months_ago_sales)
product_price = st.sidebar.number_input("💲 Product Price ($)", min_value=0.0, value=product_price)
product_star_rating = st.sidebar.slider("⭐ Product Star Rating", min_value=0.0, max_value=5.0, value=product_star_rating)
product_num_ratings = st.sidebar.number_input("📢 Number of Ratings", min_value=0, value=product_num_ratings)
product_num_offers = st.sidebar.number_input("🏷️ Number of Offers", min_value=0, value=product_num_offers)
is_best_seller = st.sidebar.selectbox("🔥 Is Best Seller?", [0, 1], index=is_best_seller)
is_prime = st.sidebar.selectbox("🚀 Is Prime?", [0, 1], index=is_prime)

# Collect user input
user_input = [
    previous_month_sales, 
    two_months_ago_sales, 
    product_price, 
    product_star_rating, 
    product_num_ratings, 
    product_num_offers, 
    is_best_seller, 
    is_prime
]

# Function to preprocess input before prediction
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_list)
    input_data_scaled = scaler.transform(input_df)
    return input_data_scaled

# Predict button
if st.sidebar.button("🔍 Predict Sales"):
    processed_input = preprocess_input(user_input)
    predicted_sales = model.predict(processed_input)[0]
    st.success(f"📊 **Predicted Sales for Next Month:** {int(predicted_sales)} units")

st.sidebar.markdown("---")
st.sidebar.info("🔹 Start typing a product title to see matching options and autofill product details.")
