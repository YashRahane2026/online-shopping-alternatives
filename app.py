import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Shopping Alternatives", page_icon="🛍️")
st.title("Online Shopping Alternatives Recommendation System")
st.write("Choose a product to find similar or cheaper alternatives.")


# Step 1: Read and prepare the Amazon CSV file
@st.cache_data
def prepare_data():
    file = Path(__file__).with_name("amazon.csv")
    df = pd.read_csv(
        file,
        usecols=["product_name", "category", "discounted_price", "about_product", "product_link"],
    )

    # Convert prices like ₹1,099 into numbers like 1099
    df["price"] = pd.to_numeric(
        df["discounted_price"].astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["product_name", "price"]).drop_duplicates("product_name")
    df = df.reset_index(drop=True)

    # Keep the final part of the category path, e.g. USBCables
    df["category"] = df["category"].fillna("Other").astype(str).str.split("|", regex=False).str[-1]
    df["about_product"] = df["about_product"].fillna("").astype(str)
    df["product_link"] = df["product_link"].fillna("").astype(str)

    # Step 2: Combine product names, categories, and descriptions
    df["text"] = df["product_name"] + " " + df["category"] + " " + df["about_product"]

    # Step 3: Convert text to numbers using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(df["text"])
    return df, vectors


products, vectors = prepare_data()

# Step 4: Let the user select a product
selected_id = st.selectbox(
    "Select a product",
    products.index,
    format_func=lambda i: f"{products.loc[i, 'product_name']} — ₹{products.loc[i, 'price']:,.0f}",
)
selected = products.loc[selected_id]
st.write(f"**Selected product:** {selected['product_name']}")
st.write(f"**Category:** {selected['category']} | **Price:** ₹{selected['price']:,.0f}")

cheaper_only = st.checkbox("Show only cheaper alternatives")
top_n = st.slider("Number of recommendations", 1, 10, 5)

# Step 5: Compare products using cosine similarity
scores = cosine_similarity(vectors[selected_id], vectors).flatten()

# Prefer products from the same category
candidates = products.index[products["category"] == selected["category"]].tolist()
candidates = [i for i in candidates if i != selected_id]
if not candidates:  # If the category has no alternatives, search all products
    candidates = [i for i in products.index if i != selected_id]

# Optional price filter
if cheaper_only:
    candidates = [i for i in candidates if products.loc[i, "price"] < selected["price"]]

# Step 6: Rank products from most similar to least similar
candidates = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_n]

# Step 7: Show the recommendations
st.subheader("Recommended Alternatives")
if not candidates:
    st.info("No alternatives found. Try turning off the cheaper-only filter.")
else:
    result = products.loc[candidates, ["product_name", "category", "price", "product_link"]].copy()
    result["Similarity (%)"] = [round(scores[i] * 100, 2) for i in candidates]
    result = result.rename(columns={
        "product_name": "Product",
        "category": "Category",
        "price": "Price (₹)",
        "product_link": "Product Link",
    })
    st.dataframe(
        result,
        hide_index=True,
        use_container_width=True,
        column_config={"Product Link": st.column_config.LinkColumn("Product Link", display_text="Open product")},
    )

st.caption("Similarity is based on product text. Dataset prices may not be current.")
