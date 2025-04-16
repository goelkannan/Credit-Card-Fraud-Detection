# Credit Card Fraud Detection using Machine Learning

## 🧠 Problem Statement

The goal of this project is to develop machine learning models that can detect fraudulent credit card transactions with high accuracy.

We will be using a dataset from a research collaboration between **Worldline** and the **Machine Learning Group**, available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset contains **284,807 transactions**, of which only **492** are fraudulent, making it highly imbalanced. Handling this imbalance is crucial before model building.

---

## 🏦 Business Context

For financial institutions, especially banks, retaining high-value customers is a top priority. However, **credit card fraud** is a major threat to both profitability and customer trust.

According to the **Nilson Report**, credit card fraud was expected to reach **$30 billion globally by 2020**. As digital payment methods grow, fraudulent transactions have become more frequent and complex.

**Machine learning** offers a scalable solution to detect fraud efficiently and proactively, reducing:

- Manual reviews
- Chargebacks and fees
- False declines of genuine transactions

---

## 🔍 What is Credit Card Fraud?

Credit card fraud involves unauthorized use of credit card information to make purchases or withdraw funds. Common types include:

- **Skimming** – Copying data from the magnetic stripe
- **Card manipulation or alteration**
- **Counterfeit cards**
- **Stolen or lost cards**
- **Telemarketing fraud**

---

## 📊 Dataset Overview

- **Total Transactions**: 284,807  
- **Fraudulent Transactions**: 492 (≈0.172%)  
- **Duration**: 2 days in September 2013  
- **Source**: European cardholders  

### Features:

| Feature | Description |
|--------|-------------|
| `Time` | Seconds elapsed between each transaction and the first transaction in the dataset |
| `Amount` | Transaction amount |
| `V1` - `V28` | Principal Components obtained via PCA to preserve confidentiality |
| `Class` | Target variable (0 = Legitimate, 1 = Fraud) |

---

## 🔄 Project Pipeline

### 1. 🧾 Data Understanding

- Load the dataset
- Inspect feature types, missing values, and class imbalance
- Understand the relationship between variables and target class

### 2. 📈 Exploratory Data Analysis (EDA)

- Perform univariate and bivariate analysis
- Check for skewness in numerical features
- Visualize the imbalance between fraud and non-fraud classes
- No need for Z-scaling due to PCA, but scaling `Amount` and `Time` might be useful

### 3. 🧪 Train/Test Split

- Split the dataset into training and testing sets
- Use **k-fold cross-validation** to ensure reliable validation and representation of minority class in each fold

### 4. 🤖 Model Building & Hyperparameter Tuning

- Try multiple classifiers:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SVM
  - KNN
- Apply techniques to handle imbalance:
  - **SMOTE (Synthetic Minority Oversampling Technique)**
  - **Random Undersampling**
- Tune hyperparameters using **GridSearchCV** or **RandomizedSearchCV**

### 5. ✅ Model Evaluation

Given the imbalanced dataset, accuracy is not a reliable metric. Instead, use:

- **Precision** – How many predicted frauds are actually fraud
- **Recall** – How many actual frauds were correctly predicted
- **F1 Score** – Harmonic mean of precision and recall
- **ROC-AUC** – Tradeoff between TPR and FPR

---

