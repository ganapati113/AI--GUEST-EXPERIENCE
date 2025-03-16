import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# MongoDB Connection
client = MongoClient("mongodb+srv://mounika:Mounika123@newdata.to1xh.mongodb.net/?retryWrites=true&w=majority&appName=newdata")
db = client["hotel_data"]

# Load Data
booking_df = pd.DataFrame(list(db["booking_data"].find()))
dining_df = pd.DataFrame(list(db["dining_info"].find()))
reviews_df = pd.DataFrame(list(db["reviews_data"].find()))

# Data Cleaning
for df in [booking_df, dining_df, reviews_df]:
    df.drop(columns=["_id"], errors="ignore", inplace=True)

booking_df["check_in_date"] = pd.to_datetime(booking_df["check_in_date"], errors="coerce")
booking_df["check_out_date"] = pd.to_datetime(booking_df["check_out_date"], errors="coerce")
booking_df["length_of_stay"] = (booking_df["check_out_date"] - booking_df["check_in_date"]).dt.days
booking_df["week"] = booking_df["check_in_date"].dt.strftime('%Y-%U')

dining_df["order_time"] = pd.to_datetime(dining_df["order_time"], errors="coerce")
dining_df["date"] = dining_df["order_time"].dt.date

reviews_df["Rating"] = pd.to_numeric(reviews_df["Rating"], errors="coerce")

# Sidebar Filters
st.sidebar.title("Filters")
start_date = st.sidebar.date_input("Start Date", booking_df["check_in_date"].min())
end_date = st.sidebar.date_input("End Date", booking_df["check_in_date"].max())

filtered_booking_df = booking_df[(booking_df["check_in_date"] >= pd.to_datetime(start_date)) & (booking_df["check_in_date"] <= pd.to_datetime(end_date))]
filtered_dining_df = dining_df[(dining_df["order_time"] >= pd.to_datetime(start_date)) & (dining_df["order_time"] <= pd.to_datetime(end_date))]
filtered_reviews_df = reviews_df

# Sidebar Navigation
option = st.sidebar.radio("Go to", ["Hotel Booking Insights", "Dining Insights", "Reviews Analysis"])

# Hotel Booking Insights
if option == "Hotel Booking Insights":
    st.title("🏨 Hotel Booking Insights")
    
    # Bookings Trend Over Time
    st.subheader("📈 Bookings Trend Over Time")
    bookings_trend = filtered_booking_df.groupby("check_in_date").size().reset_index(name="bookings")
    fig_bookings = px.line(bookings_trend, x="check_in_date", y="bookings", title="Hotel Bookings Over Time")
    st.plotly_chart(fig_bookings)

    # Preferred Cuisine Analysis
    st.subheader("🍽️ Preferred Cuisine Analysis")
    cuisine_filter = st.multiselect("Select Cuisine", booking_df["Preferred Cusine"].dropna().unique())
    if cuisine_filter:
        filtered_booking_df = filtered_booking_df[filtered_booking_df["Preferred Cusine"].isin(cuisine_filter)]
    if "Preferred Cusine" in filtered_booking_df.columns:
        fig_cuisine = px.pie(filtered_booking_df, names="Preferred Cusine", title="Preferred Cuisine Distribution")
        st.plotly_chart(fig_cuisine)
    else:
        st.warning("No cuisine data available.")

    # Average Length of Stay
    st.subheader("🛏️ Average Length of Stay")
    avg_stay = filtered_booking_df.groupby("week")["length_of_stay"].mean().reset_index()
    fig_stay = px.line(avg_stay, x="week", y="length_of_stay", title="Weekly Average Length of Stay")
    st.plotly_chart(fig_stay)

# Dining Insights
elif option == "Dining Insights":
    st.title("🍴 Dining Insights")
    
    # Average Dining Cost by Cuisine
    st.subheader("💰 Average Dining Cost by Cuisine")
    if "Preferred Cusine" in filtered_dining_df.columns and "price_for_1" in filtered_dining_df.columns:
        fig_dining_cost = px.pie(filtered_dining_df, names="Preferred Cusine", values="price_for_1", title="Dining Cost Distribution")
        st.plotly_chart(fig_dining_cost)
    else:
        st.warning("No dining cost data available.")
    
    # Customer Count Over Time
    st.subheader("👥 Customer Count Over Time")
    customer_count = filtered_dining_df.groupby("date")["customer_id"].nunique().reset_index()
    fig_customer = px.line(customer_count, x="date", y="customer_id", title="Customer Visits Over Time")
    st.plotly_chart(fig_customer)

# Reviews Analysis
elif option == "Reviews Analysis":
    st.title("📝 Reviews Analysis")
    
    # Sentiment Analysis Pie Chart
    st.subheader("📊 Sentiment Analysis")
    if "sentiment_score" in filtered_reviews_df.columns:
        fig_sentiment = px.pie(filtered_reviews_df, names="sentiment_score", title="Customer Sentiment Distribution")
        st.plotly_chart(fig_sentiment)
    else:
        st.warning("No sentiment data available.")
    
    # Rating Distribution Filter
    st.subheader("⭐ Rating Distribution")
    rating_filter = st.slider("Select Rating Range", min_value=1, max_value=5, value=(1, 5))
    filtered_reviews_df = filtered_reviews_df[(filtered_reviews_df["Rating"] >= rating_filter[0]) & (filtered_reviews_df["Rating"] <= rating_filter[1])]
    fig_ratings = px.histogram(filtered_reviews_df, x="Rating", title="Hotel Rating Distribution", nbins=5)
    st.plotly_chart(fig_ratings)
    
    # Word Cloud of Customer Feedback
    st.subheader("🗨️ Word Cloud of Customer Feedback")
    if "Review" in filtered_reviews_df.columns:
        text = " ".join(filtered_reviews_df["Review"].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        st.image(buffer, caption="Word Cloud of Reviews")
    else:
        st.warning("No review text data available.")
