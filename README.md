# Customer-Churn-Prediction-app
Customer churn prediction using Machine Learning and Streamlit

# Customer Churn Prediction System

A Machine Learning web application that predicts whether a telecom customer is likely to churn or stay.  
The application is built using Python, Scikit-learn, and Streamlit and deployed on Streamlit Cloud.

---

## Live Application
Streamlit App: https://your-streamlit-app-link  
GitHub Repository: https://github.com/yourusername/customer-churn-prediction

---

## Project Overview
Customer churn is a major challenge for telecom companies. Retaining existing customers is more cost-effective than acquiring new ones.  
This project builds a predictive machine learning model that identifies customers who are at risk of leaving the service.

The application allows users to input customer details and receive:

- Churn prediction
- Churn probability
- Risk indicator (High / Medium / Low)
- Feature importance visualization
- Customer input summary

---

## Features
- Machine Learning churn prediction model
- Interactive Streamlit web interface
- Churn probability visualization
- Feature importance chart
- Risk level indicator (High, Medium, Low)
- Transparent UI with background visualization
- Customer input summary table

---

## Technologies Used

| Technology | Purpose |
|------------|--------|
Python | Core programming language |
Pandas | Data manipulation |
NumPy | Numerical operations |
Scikit-learn | Machine learning model |
Matplotlib | Data visualization |
Streamlit | Web application framework |
Joblib | Model serialization |

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Deployment using Streamlit

---

## Model Used
Random Forest Classifier

Why Random Forest?
- Handles non-linear relationships
- Works well with tabular datasets
- Provides feature importance
- Robust against overfitting

---

## Project Structure
customer-churn-prediction
│
├── app.py
├── models
│ └── churn_model.pkl
│
├── data
│ └── churn_dataset.csv
│
├── images
│ └── churn_bg.png
│
├── requirements.txt
└── README.md
