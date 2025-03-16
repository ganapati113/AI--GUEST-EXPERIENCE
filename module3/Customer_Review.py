import streamlit as st
import pandas as pd
import os
from datetime import datetime

def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    else:
        return pd.DataFrame(columns=["review_id", "customer_id", "review_date", "Review", "Rating", "review_date_numeric", "currently_staying"])

def save_data(data, file_path):
    data.to_excel(file_path, index=False)

def main():
    file_path = "reviews_data.xlsx"
    st.title("📢 Customer Review Submission Form 📝")
    
    review_id = st.text_input("🔢 Review ID")
    customer_id = st.text_input("🆔 Customer ID")
    review_text = st.text_area("💬 Review")
    rating = st.slider("⭐ Rating", 1, 10, 3)
    currently_staying = st.radio("🏨 Are you currently staying?", ("Yes", "No"))
    
    if st.button("✅ Submit Review"):
        if review_id and customer_id and review_text:
            review_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            review_date_numeric = datetime.now().timestamp()
            
            new_data = pd.DataFrame({
                "review_id": [review_id],
                "customer_id": [customer_id],
                "review_date": [review_date],
                "Review": [review_text],
                "Rating": [rating],
                "review_date_numeric": [review_date_numeric],
                "currently_staying": [currently_staying]
            })
            
            data = load_data(file_path)
            
            if currently_staying == "Yes":
                data = pd.concat([data, new_data], ignore_index=True)
            else:
                data = pd.concat([new_data, data], ignore_index=True)
            
            save_data(data, file_path)
            st.success("🎉 Review submitted successfully!")
            
            st.subheader("📊 Updated Reviews Data")
            st.dataframe(data)
        else:
            st.error("⚠️ Please fill all the required fields.")

if __name__ == "__main__":
    main()
