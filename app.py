"""Online Shopping Alternatives - beginner-friendly IR mini project."""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Page setup
st.set_page_config(page_title="Shopping Alternatives", page_icon="🛍️", layout="wide")
st.title("🛍️ Online Shopping Alternatives")
st.write("Choose a product to find similar alternatives from the Amazon dataset.")


# Step 1: Load and prepare the product dataset
@st.cache_data
def load_products():
    file_path = Path(__file__).with_name("amazon.csv")
    products = pd.read_csv(file_path)

    # Convert prices like "₹1,299" into numbers like 1299
    products["price"] = (
        products["discounted_price"].astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    products["price"] = pd.to_numeric(products["price"], errors="coerce")

    # Remove rows that do not have a product name or price
    products = products.dropna(subset=["product_name", "price"])
    products = products.drop_duplicates(subset="product_name").reset_index(drop=True)

    # Keep the final category name, e.g. "USBCables"
    products["category"] = (
        products["category"].fillna("Other").astype(str)
        .str.split("|", regex=False).str[-1].str.strip()
    )
    products["about_product"] = products["about_product"].fillna("").astype(str)
    products["product_link"] = products["product_link"].fillna("").astype(str)

    # Combine text fields that help us compare products
    products["search_text"] = (
        products["product_name"].astype(str) + " "
        + products["category"] + " "
        + products["about_product"]
    )
    return products


# Step 2: Convert product text into TF-IDF vectors
@st.cache_data
def create_vectors(product_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer.fit_transform(product_text)


products = load_products()
tfidf_vectors = create_vectors(tuple(products["search_text"]))


# Step 3: Let the user choose a product
selected_index = st.selectbox(
    "Select a product",
    options=products.index.tolist(),
    format_func=lambda i: f"{products.at[i, 'product_name']} — ₹{products.at[i, 'price']:,.0f}",
)
selected = products.loc[selected_index]

# Show the selected product without relying on external product images
with st.container(border=True):
    st.caption("SELECTED PRODUCT")
    st.subheader(selected["product_name"])
    details_col, price_col = st.columns([4, 1])
    with details_col:
        st.write(f"Category: {selected['category']}")
    with price_col:
        st.metric("Price", f"₹{selected['price']:,.0f}")


# Step 4: Let the user choose recommendation options
option_col, count_col = st.columns([2, 1])
with option_col:
    cheaper_only = st.checkbox("Show only cheaper alternatives")
with count_col:
    result_count = st.selectbox("Number of results", [3, 5, 8, 10], index=1)


# Step 5: Find and rank similar products
scores = cosine_similarity(tfidf_vectors[selected_index], tfidf_vectors).flatten()

# Prefer products in the same category. If none exist, compare all products.
candidates = [
    i for i in products.index
    if i != selected_index and products.at[i, "category"] == selected["category"]
]
if not candidates:
    candidates = [i for i in products.index if i != selected_index]

# Keep only products priced lower than the selected product, if requested
if cheaper_only:
    candidates = [i for i in candidates if products.at[i, "price"] < selected["price"]]

# Highest cosine-similarity score comes first
recommendations = sorted(candidates, key=lambda i: scores[i], reverse=True)[:result_count]


# Step 6: Display recommendations as simple text cards (no product images)
st.subheader("Recommended alternatives")

if not recommendations:
    st.info("No alternatives match these options. Try turning off the cheaper-only filter.")
else:
    card_columns = st.columns(3)

    for position, product_index in enumerate(recommendations):
        product = products.loc[product_index]
        similarity_percent = scores[product_index] * 100
        price_difference = selected["price"] - product["price"]

        with card_columns[position % 3]:
            with st.container(border=True):
                st.markdown(f"**{product['product_name']}**")
                st.caption(product["category"])
                st.subheader(f"₹{product['price']:,.0f}")
                st.caption(f"Text similarity: {similarity_percent:.1f}%")

                if price_difference > 0:
                    st.success(f"Save ₹{price_difference:,.0f}")
                else:
                    st.write("Similar alternative")

                # Show a link only when the dataset contains a valid web URL
                link = product["product_link"].strip()
                if link.startswith(("https://", "http://")):
                    st.markdown(f"[View product ↗]({link})")


# Simple explanation for the lab presentation
with st.expander("How does this recommendation system work?"):
    st.write("1. Read product names, categories, descriptions and prices from the CSV file.")
    st.write("2. Use TF-IDF to convert product text into numerical vectors.")
    st.write("3. Use cosine similarity to compare the selected product with other products.")
    st.write("4. Optionally filter out products that are not cheaper.")
    st.write("5. Sort by similarity and display the top results.")

st.caption("Similarity measures how alike the product text is; it does not measure product quality. Dataset prices may not be current.")
