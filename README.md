# 🛡️ Credit Card Fraud Detection Dashboard

A real-time, interactive dashboard built with Python and Tkinter to predict and visualize fraudulent credit card transactions using machine learning (Logistic Regression). It features a user-friendly GUI, dynamic risk threshold adjustment, interactive charts, and live transaction logging.

## 🚀 Features

- **Logistic Regression-based Prediction**
- **SMOTE for Imbalanced Data Handling**
- **Scalable GUI** with scrollable input form
- **Interactive Threshold Slider** for customizing fraud sensitivity
- **Real-time Charts** (Pie & Line) with Matplotlib
- **Transaction Log Viewer** with the latest 20 records
- **Alerts for High-Risk Transactions**

---

## 📸 Screenshots

<img src="screenshots/dashboard.png" width="600"/>
*(Add a screenshot of your dashboard if possible)*

---

## 🧠 How It Works

1. Loads the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Preprocesses features:
   - Scales `Time` and `Amount`
   - Applies `SMOTE` to oversample minority class
3. Trains a `LogisticRegression` model
4. Accepts transaction inputs via GUI
5. Predicts fraud probability in real-time and updates:
   - Decision history table
   - Risk trend line chart
   - Fraud vs Legit pie chart

---

## 🧰 Technologies Used

- Python 3.x
- Tkinter
- Pandas, NumPy
- Scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib
- ttk Treeview

---


