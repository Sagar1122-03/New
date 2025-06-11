import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Page Config ---
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4B8BBE;'>📞 Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Choose a section", ["📤 Upload & Overview", "⚙️ Preprocessing", "📊 Model Evaluation", "📈 Visualization", "➕ Add New Data"])

# --- Data Loading ---
DATA_URL = "https://raw.githubusercontent.com/Sagar1122-03/New/refs/heads/main/Telco.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df.dropna()

df_raw = load_data()

# --- Preprocessing ---
df = df_raw.copy()
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# --- Section Routing ---
if page == "📤 Upload & Overview":
    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df_raw.head())
    st.success("✅ Data loaded and cleaned successfully!")

elif page == "⚙️ Preprocessing":
    st.subheader("🔍 After Preprocessing")
    st.dataframe(df.head())
    st.write("📌 Feature Columns:", X.columns.tolist())

elif page == "📊 Model Evaluation":
    st.subheader("🧠 Logistic Regression Evaluation")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
        st.subheader("📋 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("🧩 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        st.pyplot(fig)

elif page == "📈 Visualization":
    st.subheader("📊 Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df_raw, palette="pastel", ax=ax)
    st.pyplot(fig)

elif page == "➕ Add New Data":
    st.subheader("➕ Add New Customer for Prediction")

    with st.form("new_data_form"):
        customerID = st.text_input("Customer ID")
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 100, 12)
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
        Churn = st.selectbox("Actual Churn (Optional)", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Churn")

        if submitted:
            new_data = pd.DataFrame([{
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
                'TotalCharges': TotalCharges
            }])

            # Preprocess new data
            new_data_encoded = pd.get_dummies(new_data)
            new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)
            new_data_scaled = scaler.transform(new_data_encoded)

            # Predict
            prediction = model.predict(new_data_scaled)[0]
            prob = model.predict_proba(new_data_scaled)[0][prediction]

            st.markdown("---")
            st.success(f"🎯 **Predicted Churn:** {'Yes' if prediction == 1 else 'No'}")
            st.info(f"📈 Probability: `{prob*100:.2f}%`")
