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
page = st.sidebar.radio("Go to", ["Upload & Overview", "Preprocessing", "Model Evaluation", "Visualization", "Add New Data"])

# Title
st.title("Telco Customer Churn Prediction")

# Load Dataset
uploaded_file = "https://raw.githubusercontent.com/Sagar1122-03/New/refs/heads/main/Telco.csv"

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    df_raw.dropna(inplace=True)

    df = df_raw.copy()
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)

    # Features and Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Page Views
    if page == "Upload & Overview":
        st.subheader("Raw Data Preview")
        st.write(df_raw.head())
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
        sns.countplot(x="Churn", data=df_raw, ax=ax2)
        st.pyplot(fig2)

    elif page == "Add New Data":
        st.subheader("Add New Customer Record")

        with st.form("input_form"):
            customerID = st.text_input("Customer ID")
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure", min_value=0, max_value=100)
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0)
            Churn = st.selectbox("Churn", ["Yes", "No"])

            submitted = st.form_submit_button("Submit")

            if submitted:
                # Create dataframe for single input
                new_data = pd.DataFrame([{
                    'customerID': customerID,
                    'gender': gender,
                    'SeniorCitizen': SeniorCitizen,
                    'Partner': Partner,
                    'Dependents': Dependents,
                    'tenure': tenure,
                    'PhoneService': PhoneService,
                    'MultipleLines': MultipleLines,
                    'InternetService': InternetService,
                    'OnlineSecurity': OnlineSecurity,
                    'OnlineBackup': OnlineBackup,
                    'DeviceProtection': DeviceProtection,
                    'TechSupport': TechSupport,
                    'StreamingTV': StreamingTV,
                    'StreamingMovies': StreamingMovies,
                    'Contract': Contract,
                    'PaperlessBilling': PaperlessBilling,
                    'PaymentMethod': PaymentMethod,
                    'MonthlyCharges': MonthlyCharges,
                    'TotalCharges': TotalCharges,
                    'Churn': 1 if Churn == "Yes" else 0
                }])

                # Drop customerID
                new_data_processed = new_data.drop(['customerID'], axis=1)

                # Encode and match training format
                new_data_encoded = pd.get_dummies(new_data_processed)
                # Align to training data columns
                new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)

                # Scale
                new_data_scaled = scaler.transform(new_data_encoded)

                # Predict
                prediction = model.predict(new_data_scaled)[0]
                st.success(f"Predicted Churn: {'Yes' if prediction == 1 else 'No'}")
else:
    st.warning("Please upload a Telco CSV file to continue.")
