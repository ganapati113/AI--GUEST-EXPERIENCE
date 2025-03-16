from pymongo import MongoClient
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

# Connect to MongoDB
client = MongoClient("mongodb+srv://mounika:Mounika123@newdata.to1xh.mongodb.net/?retryWrites=true&w=majority&appName=newdata")
db = client["hotel_guests"]
collection = db["dining_info"]

# Load data from MongoDB
df_from_mongo = pd.DataFrame(list(collection.find()))
df = df_from_mongo.copy()

# Check if data is loaded properly
if df.empty:
    print("Error: No data found in MongoDB.")
    exit()

# Convert date columns to datetime
df['check_in_date'] = pd.to_datetime(df['check_in_date'], errors='coerce')
df['check_out_date'] = pd.to_datetime(df['check_out_date'], errors='coerce')
df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')

# Drop rows where essential dates are missing
df.dropna(subset=['check_in_date', 'check_out_date', 'order_time'], inplace=True)

# Feature engineering
df['check_in_day'] = df['check_in_date'].dt.dayofweek
df['check_out_day'] = df['check_out_date'].dt.dayofweek
df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days

# Time-based features
df['order_hour'] = df['order_time'].dt.hour
df['order_day'] = df['order_time'].dt.day
df['order_month'] = df['order_time'].dt.month

# Customer loyalty feature
df['days_since_first_order'] = (df['order_time'] - df.groupby('customer_id')['order_time'].transform('min')).dt.days

# Dish popularity
dish_popularity = df.groupby('dish').size().reset_index(name='dish_popularity')
df = df.merge(dish_popularity, on='dish', how='left')

# Cuisine popularity
cuisine_popularity = df.groupby('Preferred Cusine').size().reset_index(name='cuisine_popularity')
df = df.merge(cuisine_popularity, on='Preferred Cusine', how='left')

# Price per quantity
df['price_per_qty'] = df['price_for_1'] / df['Qty']

# Splitting dataset
train_df = df[df['order_time'] < '2024-10-01']
test_df = df[df['order_time'] >= '2024-10-01']

# Customer-level aggregations
customer_features = train_df.groupby('customer_id').agg(
    total_orders_per_customer=('transaction_id', 'count'),
    avg_spend_per_customer=('price_for_1', 'mean'),
    total_qty_per_customer=('Qty', 'sum'),
    avg_stay_per_customer=('stay_duration', 'mean'),
    avg_days_since_first_order=('days_since_first_order', 'mean')
).reset_index()

# Get most frequent cuisine & dish per customer
customer_dish = train_df.groupby('customer_id').agg(
    most_frequent_dish=('dish', lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
).reset_index()

# Cuisine-level aggregations
cuisine_features = train_df.groupby('Preferred Cusine').agg(
    avg_price_per_cuisine=('price_for_1', 'mean'),
    total_orders_per_cuisine=('transaction_id', 'count')
).reset_index()

# Most popular dish per cuisine
cuisine_dish = train_df.groupby('Preferred Cusine').agg(
    cuisine_popular_dish=('dish', lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
).reset_index()

# Merge features into train and test data
train_df = train_df.merge(customer_features, on='customer_id', how='left')
train_df = train_df.merge(cuisine_features, on='Preferred Cusine', how='left')
train_df = train_df.merge(customer_dish, on='customer_id', how='left')
train_df = train_df.merge(cuisine_dish, on='Preferred Cusine', how='left')

test_df = test_df.merge(customer_features, on='customer_id', how='left')
test_df = test_df.merge(cuisine_features, on='Preferred Cusine', how='left')
test_df = test_df.merge(customer_dish, on='customer_id', how='left')
test_df = test_df.merge(cuisine_dish, on='Preferred Cusine', how='left')

# Fill missing categorical values
for col in ['most_frequent_dish', 'cuisine_popular_dish']:
    train_df[col].fillna("Unknown", inplace=True)
    test_df[col].fillna("Unknown", inplace=True)

# Drop unnecessary columns
columns_to_drop = ['_id', 'transaction_id', 'customer_id', 'price_for_1', 'Qty', 'order_time', 'check_in_date', 'check_out_date']
train_df.drop(columns_to_drop, axis=1, inplace=True)
test_df.drop(columns_to_drop, axis=1, inplace=True)

# One-hot encoding for categorical columns
categorical_cols = ['Preferred Cusine', 'most_frequent_dish', 'cuisine_popular_dish']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_array = encoder.fit_transform(train_df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
train_df = pd.concat([train_df.drop(columns=categorical_cols), encoded_df], axis=1)

# Save the encoder
joblib.dump(encoder, 'encoder.pkl')

# Apply encoding to test data
encoded_test = encoder.transform(test_df[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
test_df = pd.concat([test_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

# Encode the target column 'dish'
label_encoder = LabelEncoder()
train_df['dish'] = label_encoder.fit_transform(train_df['dish'])
test_df['dish'] = label_encoder.transform(test_df['dish'])

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split into features (X) and target (y)
X_train = train_df.drop(columns=['dish'])
y_train = train_df['dish']
X_test = test_df.drop(columns=['dish'])
y_test = test_df['dish']

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    learning_rate=0.8,
    max_depth=6,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(xgb_model, 'xgb_model_dining.pkl')

# Evaluate the model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, xgb_model.predict_proba(X_test))

print(f"Accuracy: {accuracy}")
print(f"Log Loss: {logloss}")

# Plot top 5 features
xgb.plot_importance(xgb_model, max_num_features=5)
plt.show()
