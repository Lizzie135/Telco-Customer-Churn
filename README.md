# Telco Customer Churn Prediction: A Machine Learning Approach

**Author:** Elizabeth-Lynda Nartey

---

## Overview
Customer churn is a major challenge in the telecommunications industry. This project develops a machine learning-based predictive model to identify customers at risk of leaving, enabling proactive retention strategies.

---

## Dataset
The project uses the [Telco Customer Churn Dataset from Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) containing 7,043 customer records with demographic, account, and service information.

**Key Features:**
- **Demographic:** Gender, SeniorCitizen, Partner, Dependents  
- **Account Information:** Tenure, Contract type, Payment method  
- **Service Information:** Phone service, Internet service, Online security, Tech support  
- **Target Variable:** Churn (1 = Yes, 0 = No)

---

## Methodology

1. **Data Understanding**
   - Initial exploration and identification of missing values
   - Class imbalance: ~26% of customers churned

2. **Data Preprocessing**
   - Converted `TotalCharges` to numeric, imputed missing values
   - Label encoded categorical variables
   - Standardized numerical features

3. **Data Exploration**
   - Visualized patterns in churn across services, contracts, and demographics
   - Key drivers identified: Contract Type, Tenure, Tech Support, Online Security, Total Charges

4. **Handling Class Imbalance**
   - Applied SMOTE to balance classes in training data
   - Compared results with and without SMOTE

5. **Modeling**
   - Logistic Regression (baseline, interpretable)
   - Random Forest Classifier (non-linear patterns)
   - Multilayer Perceptron (MLP)

6. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC
   - Random Forest achieved the best performance

---

## Experimental Results

**Best Model:** Random Forest without SMOTE  
- Accuracy: 94%  
- ROC AUC: 0.9786  
- Precision (churned customers): 0.88  
- Recall (churned customers): 0.88  

**Insights:**  
- Month-to-month contracts, lack of tech support, and fiber optic internet are associated with higher churn  
- SMOTE provided minor improvements for non-linear models

---

## Tools & Environment
- **Languages & Libraries:** Python, pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn  
- **Environment:** Jupyter Notebook  
- **Random State:** 42 (for reproducibility)

---

## Future Work
- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV  
- Advanced classifiers: XGBoost, LightGBM, CatBoost  
- Dimensionality reduction (PCA)  
- Deep learning enhancements (MLP / neural networks)  
- Time-series analysis on tenure to predict churn timing  
- Incorporating external market data for deeper insights

---

## References
- Brownlee, J. (2020). *Imbalanced Classification with Python*. Machine Learning Mastery.  
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.  
- Kaggle. *Telco Customer Churn Dataset*. https://www.kaggle.com/blastchar/telco-customer-churn  
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830

