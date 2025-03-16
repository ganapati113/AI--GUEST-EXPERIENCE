import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb+srv://mounika:Mounika123@newdata.to1xh.mongodb.net/?retryWrites=true&w=majority&appName=newdata")
db = client["newdata"]
bookings_collection = db["bookings"]
model_collection = db["models"]
orders_collection = db["orders"]  # Collection for storing past orders

# Load model from MongoDB
model_doc = model_collection.find_one({"model_name": "xgb_dish_recommendation"})
if model_doc:
    model_binary = model_doc["model_binary"]
    with open("retrieved_xgb_model.pkl", "wb") as f:
        f.write(model_binary)
    with open("retrieved_xgb_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    st.success("Model loaded successfully!")
else:
    st.error("Model not found in MongoDB!")
    st.stop()

# Streamlit UI
st.title("Hotel Booking Form")
st.write("Enter booking details.")

# Input form
name = st.text_input("Name")
customer_id = st.text_input("Customer ID")
check_in_date = st.date_input("Check-in Date")
check_out_date = st.date_input("Check-out Date")
age = st.number_input("Age", min_value=1, max_value=120, step=1)
preferred_cuisine = st.selectbox("Preferred Cuisine", ["South Indian", "North Indian", "Multi"])
special_requests = st.text_area("Any Special Requests?")
stay_duration = (check_out_date - check_in_date).days

if st.button("Submit Booking"):
    if name and customer_id and preferred_cuisine and stay_duration > 0:
        # Prepare new booking data
        new_booking = {
            "name": name,
            "customer_id": customer_id,
            "check_in_date": check_in_date.strftime("%Y-%m-%d"),
            "check_out_date": check_out_date.strftime("%Y-%m-%d"),
            "age": age,
            "Preferred Cuisine": preferred_cuisine,
            "special_requests": special_requests,
            "stay_duration": stay_duration
        }

        # Store booking data in MongoDB
        bookings_collection.insert_one(new_booking)
        st.success("Booking saved successfully!")

        # Display entered details
        st.subheader("Entered Booking Details:")
        for key, value in new_booking.items():
            st.write(f"**{key.replace('_', ' ').title()}**: {value}")

    else:
        st.error("Please enter all details correctly!")
