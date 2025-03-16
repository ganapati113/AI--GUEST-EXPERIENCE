import streamlit as st
import pandas as pd
import os
from pinecone import Pinecone
from langchain_together import TogetherEmbeddings
from together import Together
from textblob import TextBlob  # Import TextBlob for sentiment analysis

# 🌟 Set API keys
TOGETHER_API_KEY = "485cd21c60235270dfc7bd18dc26039203aa712b0969ff6d565180fb91a1b42a"
PINECONE_API_KEY = "pcsk_5Zr5u9_BKtodyBkPEoMs69MW5KZLcTB6crVJvLTctvfP7QBZ1DRarEfG8JFw11mcogEgsa"
PINECONE_INDEX_NAME = "hotel-reviews"

# 🔑 Set environment variables
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# 📂 Load dataset
try:
    df = pd.read_excel('reviews_data.xlsx')
    if "review_id" not in df.columns or "Review" not in df.columns:
        st.error("🚨 Error: Missing 'review_id' or 'Review' columns in the dataset.")
        st.stop()
except FileNotFoundError:
    st.error("📂 Error: The file 'reviews_data.xlsx' was not found!")
    st.stop()

# 🌎 Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Ensure Pinecone Index Exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    st.error(f"⚠️ Error: Pinecone index '{PINECONE_INDEX_NAME}' does not exist! Create it first.")
    st.stop()

# 🔗 Connect to Pinecone Index
index = pc.Index(PINECONE_INDEX_NAME)

# 🤖 Initialize Together AI Embeddings
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
client = Together()

# 🎨 Streamlit UI
st.title("🏨 Hotel Customer Sentiment Analysis 📊")
st.markdown("Analyze customer feedback with AI-powered sentiment analysis! 💬✨")

query = st.text_input("🔍 Enter a query about customer reviews:", "How is the food quality?")
start_date = st.date_input("📅 Start Date")
end_date = st.date_input("📅 End Date")
rating_filter = st.slider("⭐ Select Rating Filter", 1, 10, (1, 10))

if st.button("🚀 Analyze Sentiment"):
    try:
        query_embedding = embeddings.embed_query(query)

        # 🗓️ Convert dates to YYYYMMDD format for filtering
        start_date_str = int(start_date.strftime('%Y%m%d'))
        end_date_str = int(end_date.strftime('%Y%m%d'))

        # 🔍 Query Pinecone Index
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            filter={
                "Rating": {"$gte": rating_filter[0], "$lte": rating_filter[1]},
                "review_date": {"$gte": start_date_str, "$lte": end_date_str}
            }
        )

        # 📌 Extract Matching Results
        matches = results.get("matches", [])
        if not matches:
            st.warning("⚠️ No reviews found matching the criteria.")
        else:
            matched_ids = [int(match["metadata"].get("review_id", -1)) for match in matches if "metadata" in match]
            matched_ids = [mid for mid in matched_ids if mid != -1]  # Remove invalid IDs
            
            req_df = df[df["review_id"].isin(matched_ids)]
            concatenated_reviews = " ".join(req_df["Review"].tolist())

            # 📝 Generate Sentiment Summary
            response = client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[{"role": "user", "content": f"""
                    Briefly summarize the overall sentiment of customers based on the reviews: {concatenated_reviews},
                    and query of manager: {query}.
                    Stick to the specific query of the manager and keep it short.
                    Do not mention the name of the hotel.
                """}]
            )

            # 🛠️ Perform Sentiment Analysis
            sentiment_score = TextBlob(concatenated_reviews).sentiment.polarity  # Value between -1 to 1
            
            # 📢 Display Results
            st.subheader("📌 Sentiment Summary")
            st.success(response.choices[0].message.content)
            st.subheader("📊 Final Sentiment Score")
            st.metric(label="Sentiment Score", value=round(sentiment_score, 2), delta=None)
            st.subheader("📝 Matched Reviews")
            st.dataframe(req_df[["Review"]])
    
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
