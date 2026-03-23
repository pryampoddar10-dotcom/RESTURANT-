
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

st.set_page_config(page_title="Restaurant Intelligence", layout="wide")

st.title("🍽️ AI Restaurant Decision System")

# Load data
try:
    df = pd.read_csv("final_restaurant_dataset.csv")
    st.sidebar.success("Using default dataset")
except:
    file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
    else:
        st.stop()

# Encode
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# KPI
c1, c2, c3 = st.columns(3)
c1.metric("Customers", len(df))
c2.metric("Avg Spend", int(df["avg_spend_customer"].mean()))
c3.metric("Avg Demand", int(df["total_orders_hour"].mean()))

tabs = st.tabs(["📊 Demand","👥 Customers","🤖 Models","🛒 Basket","🚀 Strategy","🧪 Simulation","📂 Predict"])

# Demand
with tabs[0]:
    fig = px.line(df.groupby("time_slot")["total_orders_hour"].mean().reset_index(),
                  x="time_slot", y="total_orders_hour")
    st.plotly_chart(fig, use_container_width=True)

# Customers
with tabs[1]:
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_encoded["cluster"] = kmeans.fit_predict(df_encoded)
    fig = px.scatter(df_encoded, x="avg_spend_customer", y="visit_frequency", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

# Models
with tabs[2]:
    X = df_encoded.drop(columns=["discount_purchase_intent"])
    y = df_encoded["discount_purchase_intent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(accuracy_score(y_test, y_pred),2))
    col2.metric("Precision", round(precision_score(y_test, y_pred, average='weighted'),2))
    col3.metric("Recall", round(recall_score(y_test, y_pred, average='weighted'),2))
    col4.metric("F1", round(f1_score(y_test, y_pred, average='weighted'),2))

# Basket
with tabs[3]:
    basket = df['items_ordered'].str.get_dummies(sep=',')
    frequent = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=0.5)
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

# Strategy
with tabs[4]:
    st.success("Increase prices during peak")
    st.info("Target discounts to sensitive users")
    st.warning("Promote combos")

# Simulation with Revenue
with tabs[5]:
    st.subheader("🧪 Price & Revenue Simulation")

    price_change = st.slider("Price Change (%)", -20, 30, 0)
    discount = st.slider("Discount (%)", 0, 50, 10)

    base_demand = df["total_orders_hour"].mean()
    base_price = df["order_value"].mean()

    simulated_demand = base_demand * (1 - price_change/100 + discount/100)
    simulated_price = base_price * (1 + price_change/100 - discount/100)

    revenue = simulated_demand * simulated_price

    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated Demand", int(simulated_demand))
    col2.metric("Simulated Price", int(simulated_price))
    col3.metric("Estimated Revenue", int(revenue))

    if revenue > base_demand * base_price:
        st.success("Strategy increases revenue ✅")
    else:
        st.error("Strategy may reduce revenue ⚠️")

# Predict
with tabs[6]:
    new_file = st.file_uploader("Upload New Customers", type=["csv"])
    if new_file:
        new_df = pd.read_csv(new_file)
        new_enc = new_df.copy()
        for col in new_enc.select_dtypes(include='object').columns:
            new_enc[col] = le.fit_transform(new_enc[col])
        preds = clf.predict(new_enc)
        new_df["Prediction"] = preds
        st.dataframe(new_df)
