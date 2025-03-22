# 🏨 Hotel Booking Cancellation Prediction

This project was completed as part of **CPSC 330: Applied Machine Learning** at the University of British Columbia by **Zhengling Jiang**. The goal of this project was to build machine learning models to predict whether a hotel booking will be canceled using the [Hotel Reservation Cancellation dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments).

---

## 📚 Table of Contents

1. Understanding the Problem  
2. Data Splitting  
3. Exploratory Data Analysis (EDA)  
4. Feature Engineering  
5. Preprocessing & Transformations  
6. Baseline Model  
7. Logistic Regression  
8. Other Classifiers (Random Forest, XGBoost, SVC)  
9. Feature Selection  
10. Hyperparameter Optimization  
11. Feature Importance & Interpretation (SHAP, eli5)  
12. Results on Test Set  
13. Explaining Predictions  
14. Summary of Results

---

## 🔍 Project Overview

- **Objective**: Predict whether a reservation will be canceled based on booking information.
- **Target**: `booking_status` (1 = canceled, 0 = not canceled)
- **Data**: ~18,000 rows × 18 features
- **Tech stack**: `scikit-learn`, `xgboost`, `SHAP`, `eli5`, `pandas`, `matplotlib`, `seaborn`

---

## ⚙️ Key Steps

- **EDA**: Uncovered class imbalance and key feature correlations (e.g. `lead_time`, `no_of_special_requests`)
- **Feature Engineering**: Created a new feature `previous_cancellation_rate`
- **Preprocessing**: Applied custom `ColumnTransformer` with imputation, scaling, and encoding
- **Modeling**:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (best performance)
  - SVC

---

## 📈 Model Performance

| Model             | F1 Score (Validation) | F1 Score (Test) |
|------------------|------------------------|-----------------|
| Logistic Regression | 0.700                  | —               |
| Random Forest       | 0.795                  | —               |
| **XGBoost (Final)** | **0.817**              | **0.829**       |
| SVC                 | 0.764                  | —               |

---

## 🧠 Interpretation

- **Tools**: SHAP, ELI5
- **Top Features**: `lead_time`, `market_segment_type`, `no_of_special_requests`
- **Insights**: Longer lead time and more special requests increase cancellation likelihood.

---

## 🧪 Results & Conclusion

The final model, an XGBoost classifier with optimized hyperparameters, achieved an **F1 score of 0.829 on the test set**, outperforming all other models and demonstrating good generalization. This model can help hotels proactively manage reservations with a high likelihood of cancellation.
