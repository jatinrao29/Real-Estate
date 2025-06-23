#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")
st.title("🏠 Real Estate Price Estimation Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Real_Estate.csv")
    df = df.dropna()
    df = pd.get_dummies(df, columns=["location"], drop_first=True)
    return df

df = load_data()

# Preprocessing
X = df.drop(columns=["property_id", "price_lakhs"])
y = df["price_lakhs"]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_predict = model.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)

# --- Sidebar Inputs ---
st.sidebar.header("📋 Input Property Features")

sample_input = {}
for col in X.columns:
    if "location_" in col:
        sample_input[col] = st.sidebar.selectbox(f"{col}", [0, 1])
    else:
        sample_input[col] = st.sidebar.number_input(f"{col}", value=float(df[col].mean()))

# Predict button
if st.sidebar.button("Predict Price"):
    input_df = pd.DataFrame([sample_input])
    input_scaled = scaler.transform(input_df)
    predicted_price = model.predict(input_scaled)
    st.subheader("📈 Predicted Price")
    st.success(f"Predicted Price: ₹ {round(predicted_price[0], 2)} Lakhs")

# --- Metrics ---
st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", round(r2, 3))
col2.metric("RMSE", round(rmse, 2))
col3.metric("MSE", round(mse, 2))

# --- Results Table ---
st.subheader("📄 Predicted vs Actual (Sample)")
results_df = pd.DataFrame({
    "Actual Price (₹ Lakhs)": y_test.values,
    "Predicted Price (₹ Lakhs)": np.round(y_predict, 2)
})
st.dataframe(results_df.head(10))

# --- Residual Plot ---
st.subheader("🔍 Residual Analysis")
residuals = y_test - y_predict
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=y_predict, y=residuals, alpha=0.6, ax=ax)
ax.axhline(0, color="orange", linestyle="--")
ax.set_xlabel("Predicted Price (₹ Lakhs)")
ax.set_ylabel("Residuals")
ax.set_title("Predicted Price vs Residuals")
st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown("Developed by Jatin Rao | Data Analyst")

