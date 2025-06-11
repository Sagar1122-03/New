import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit Page Configuration
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Overview", "Preprocessing", "Model Evaluation", "Visualization"])

# Title
st.title("Telco Customer Churn Prediction")

# Upload File
uploaded_file = 

# Load and process data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)

    # Split and scale
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Page-specific views
    if page == "Upload & Overview":
        st.subheader("Raw Data Preview")
        st.write(df.head())
        st.success("Data loaded and processed successfully!")

    elif page == "Preprocessing":
        st.subheader("Data After Preprocessing")
        st.write(df.describe())
        st.write("Columns after encoding:")
        st.write(df.columns.tolist())

    elif page == "Model Evaluation":
        st.subheader("Model Accuracy")
        st.write(f"*Accuracy:* {accuracy:.4f}")
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'], ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    elif page == "Visualization":
        st.subheader("Churn Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Churn", data=df, ax=ax2)
        st.pyplot(fig2)

else:
    st.warning("Please upload a Telco CSV file from the sidebar to continue.")
