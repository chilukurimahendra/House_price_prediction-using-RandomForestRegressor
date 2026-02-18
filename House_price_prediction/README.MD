# 🏠 End-to-End House Price Prediction System

## 📌 Project Overview
This project predicts house prices using Machine Learning techniques.
It includes complete data preprocessing, outlier removal, model training,
evaluation, and deployment using Streamlit.

---

## 🚀 Features
- Data Cleaning & Outlier Removal (IQR Method)
- Feature Engineering
- Random Forest Regression Model
- R² Score: 0.89
- Streamlit Web Application
- Model Deployment Ready

---

## 📊 Dataset
Housing dataset with features:
- area
- bedrooms
- bathrooms
- stories
- parking
- furnishingstatus
- etc.

Target Variable: `price`

---

## 🧠 Machine Learning Workflow
1. Exploratory Data Analysis
2. Outlier Removal
3. Preprocessing Pipeline (Encoding + Scaling)
4. Model Training
5. Model Evaluation
6. Model Saving
7. Web App Integration

---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit

---

## ▶️ How To Run

### Install dependencies
pip install -r requirements.txt

### Train model
python train.py

### Run web app
streamlit run app.py
