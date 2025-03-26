import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Streamlit UI
st.title("🔍 Fraud Detection Dashboard for Auditors")
st.write("This tool helps auditors identify potentially fraudulent transactions. No technical knowledge required! Simply upload a CSV file and review the results.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Transactions CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### 📝 Data Overview")
    st.write("This is a preview of your uploaded data.")
    st.write(data.head())
    
    # Convert Transaction_Date to datetime
    if 'Transaction_Date' in data.columns:
        data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'], errors='coerce')
        data['Year'] = data['Transaction_Date'].dt.year
        data['Month'] = data['Transaction_Date'].dt.month
        data['Day'] = data['Transaction_Date'].dt.day
        data.drop(columns=['Transaction_Date'], inplace=True)
    
    # One-hot encoding for categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Convert numeric fields properly
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values
    data.fillna(data.median(), inplace=True)
    
    # Fraud Risk Classification (Using Risk Score threshold)
    if 'Risk_Score' in data.columns:
        threshold = data['Risk_Score'].quantile(0.75)  # Top 25% as potential fraud
        data['Fraud Risk'] = (data['Risk_Score'] >= threshold).astype(int)
    
    st.write("### 🔍 Processed Data Preview")
    st.write(data.head())
    
    # Risk Score Distribution Visualization
    st.write("### 📈 Risk Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['Risk_Score'], bins=20, kde=True, ax=ax, color='blue')
    ax.set_title("Distribution of Risk Scores")
    st.pyplot(fig)
    
    if 'Fraud Risk' in data.columns:
        X = data.drop(columns=['Fraud Risk', 'Customer_ID'], errors='ignore')  # Exclude non-predictive columns
        y = data['Fraud Risk']
        
        # Feature scaling: Apply only to numeric columns
        numeric_features = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        # Ensure all features are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(0, inplace=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation
        st.write("### 📊 Model Accuracy")
        st.write(f"✔ Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
        
        # Fraud Detection on Full Data
        data['Fraud Prediction'] = model.predict(X)
        
        # Summary of results
        fraud_cases = data['Fraud Prediction'].sum()
        total_cases = len(data)
        fraud_percentage = (fraud_cases / total_cases) * 100
        
        st.write("### 🚨 Fraud Analysis Summary")
        st.write(f"⚠ Total Transactions: {total_cases}")
        st.write(f"🔴 Suspected Fraudulent Transactions: {fraud_cases} ({fraud_percentage:.2f}%)")
        
        # Display all high-risk transactions
        st.write("### 🔍 High-Risk Transactions")
        st.write(data[data['Fraud Prediction'] == 1][['Customer_ID', 'Transaction_Amount', 'Fraud Prediction']])
        
        # Visualization: Fraud vs. Non-Fraud Transactions
        st.write("### 📊 Fraud vs. Safe Transactions")
        fig, ax = plt.subplots()
        sns.countplot(x=data['Fraud Prediction'], palette=['green', 'red'], ax=ax)
        ax.set_xticklabels(['Safe', 'Fraud'])
        ax.set_title("Comparison of Fraudulent and Safe Transactions")
        st.pyplot(fig)
        
        # Download Results
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Fraud Analysis Report", csv, "fraud_analysis.csv", "text/csv")
        
        st.write("### 📌 How to Interpret the Results")
        st.write("1. **High-Risk Transactions:** Transactions flagged as 'Fraud' require further investigation.")
        st.write("2. **Accuracy Score:** A higher score means the model is performing well in detecting fraudulent transactions.")
        st.write("3. **Risk Score Distribution:** Helps auditors understand how transactions are categorized.")
        st.write("4. **Download Report:** Click the download button to get a detailed fraud analysis report.")
