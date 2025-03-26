# 🚀 Fraud Detection Dashboard

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

### 🎯 Introduction

The **Fraud Detection Dashboard** is an easy-to-use tool designed to help auditors identify potentially fraudulent transactions from a dataset. This project leverages machine learning models, particularly Random Forest Classifier, to analyze transaction data and classify them based on risk scores. It provides automated fraud detection, visualization, and actionable insights, making the audit process more efficient and reducing the manual effort needed.

### 🎥 Demo

🔗 [Live Demo](#) (coming soon)  
📹 [Video Demo](#) (coming soon)

🖼️ Screenshots:

![Dashboard Screenshot](screenshot1.png)

### 💡 Inspiration

We were inspired to create this project after noticing the increasing need for automated fraud detection in financial audits. Manual detection processes are time-consuming, prone to human error, and inefficient. This project aims to make the process faster and more accurate by providing auditors with a tool that highlights high-risk transactions, enabling quicker decision-making.

### ⚙️ What It Does

- **Upload CSV Files**: Users can upload transaction data in CSV format.
- **Data Preprocessing**: Automatically processes transaction data, including handling missing values and encoding categorical variables.
- **Risk Score Classification**: Analyzes the `Risk_Score` column to flag transactions as high-risk (potential fraud).
- **Fraud Detection**: Uses a Random Forest Classifier model to predict fraudulent transactions based on the data.
- **Visualizations**: Provides visualizations for risk score distribution and a comparison between safe and fraudulent transactions.
- **Downloadable Report**: Generates and allows users to download a detailed fraud analysis report in CSV format.

### 🛠️ How We Built It

We built the project using Python, Streamlit for the front-end, and scikit-learn for machine learning. The backend logic handles data preprocessing, model training, and evaluation. Here's a brief overview of the tools and libraries used:

- **Streamlit** for building the interactive dashboard.
- **Pandas & NumPy** for data manipulation and analysis.
- **Matplotlib & Seaborn** for visualizations.
- **scikit-learn** for building and evaluating the machine learning model (Random Forest Classifier).
- **Joblib** for saving and loading the model.

### 🚧 Challenges We Faced

- **Data Quality**: Many datasets have missing values, inconsistent formats, and incorrect data types. We spent time cleaning and preprocessing the data.
- **Model Performance**: Tuning the Random Forest model to achieve good performance on the data was a challenge, requiring multiple iterations of hyperparameter optimization.
- **Integration**: Ensuring smooth integration of machine learning models with Streamlit for real-time predictions and reporting was a bit tricky but rewarding once it was completed.

### 🏃 How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo.git
    cd fraud-detection-dashboard
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the project**:
    ```bash
    streamlit run test.py
    ```

### 🏗️ Tech Stack

🔹 **Frontend**: Streamlit  
🔹 **Backend**: Python  
🔹 **Machine Learning**: scikit-learn (Random Forest Classifier)  
🔹 **Data Analysis**: Pandas, NumPy  
🔹 **Visualization**: Matplotlib, Seaborn

### 👥 Team

- **Arnav Singh Rana** - [GitHub](https://github.com/your-username) | [LinkedIn](https://www.linkedin.com/in/arnavsinghrana/)
- **Akshat Srivastava** - [GitHub](https://github.com/your-username) | [LinkedIn](https://www.linkedin.com/in/akshat-srivastava-10ab75241/)
- **Sahil Jamal** - [GitHub](https://github.com/your-username) | [LinkedIn](https://www.linkedin.com/in/sahiljamalsiddiqui/)
- **Kanishk Sharma** - [GitHub](https://github.com/your-username) | [LinkedIn](https://www.linkedin.com/in/kanishk-sharma08/)

