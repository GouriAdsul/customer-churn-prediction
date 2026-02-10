📊 Customer Churn Prediction using Machine Learning
🧠 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. In this project, a telecom customer churn dataset is analyzed and multiple machine learning classification models are built to predict whether a customer will churn or not.

This repository contains a complete, step-by-step implementation starting from data loading to model evaluation, written in a beginner-friendly way.

🎯 Project Objectives

Understand customer churn using Exploratory Data Analysis (EDA)

Clean and preprocess real-world telecom data

Convert categorical data into numerical format

Train multiple machine learning models

Compare models using train accuracy, test accuracy, and cross-validation

Identify the best-performing model

📁 Dataset Information

The dataset contains customer demographic details, service subscriptions, billing information, and churn status.

🔑 Features
['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
 'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
🎯 Target Variable

Churn → Yes / No

🔍 Exploratory Data Analysis (EDA)

The following EDA steps were performed in the notebook:

Checked dataset shape and data types

Identified categorical and continuous features

Analyzed churn distribution using value_counts()

Used describe() to understand data statistics

Visualized customer churn patterns using seaborn and matplotlib

🛠 Data Preprocessing Steps

Removed unnecessary column: customerID

Separated categorical and numerical features

Encoded categorical variables into numerical format

Handled data type inconsistencies

Prepared final feature matrix (X) and target variable (y)

✂️ Train–Test Split

Dataset split into training and testing sets

Ensured unbiased model evaluation

🤖 Machine Learning Models Implemented

The following algorithms were trained and tested:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree Classifier

Random Forest Classifier

Each model was evaluated using:

Training Accuracy

Testing Accuracy

Cross-Validation Score

📈 Model Performance Comparison

📈 Model Performance Comparison

<img width="454" height="141" alt="image" src="https://github.com/user-attachments/assets/7caa6814-962f-4ad1-8f25-16444ee3e25d" />


🏆 Best Performing Model

K-Nearest Neighbors (KNN) achieved the highest test accuracy and cross-validation score

Decision Tree and Random Forest showed high training accuracy, indicating possible overfitting

⚠️ Key Observations

Tree-based models tend to overfit on training data

Logistic Regression and SVM struggled with complex patterns

Cross-validation helped confirm model stability

🧪 Evaluation Metrics Used

Accuracy Score

Cross-Validation Score

Train vs Test Accuracy Comparison

🧰 Tools & Technologies

Python 🐍

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/customer-churn-prediction.git

Install required libraries

pip install numpy pandas matplotlib seaborn scikit-learn

Open churn_prediction.ipynb

Run all cells sequentially

📌 Future Enhancements

Hyperparameter tuning

Feature importance analysis

Handling class imbalance using SMOTE

Model deployment using Flask or Streamlit

👨‍🎓 Author

Rushikesh Potdar
BBA (3rd Year) | Beginner in Machine Learning & Data Analytics

⭐ Acknowledgement

This project was created for academic learning and practical understanding of machine learning concepts using a real-world business problem.
