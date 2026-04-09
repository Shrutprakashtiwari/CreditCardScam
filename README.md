# CreditCardScam
I used multiple models to classify whether a particular pattern is a scam or not
#  Credit Card Fraud Detection using Machine Learning

##  Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques on an imbalanced dataset.
The goal is to maximize fraud detection (**recall**) while maintaining high reliability (**precision**) to reduce false alarms.

---

##  Problem Statement

Credit card fraud datasets are highly imbalanced, where fraudulent transactions form a very small percentage of total data.
Traditional accuracy-based models fail in such scenarios, so this project focuses on **precision-recall tradeoff** instead.

---

##  Approach

### 1. Data Handling

* Handled missing values using imputation
* Addressed class imbalance using:

  * `class_weight='balanced'` (Logistic Regression)
  * XGBoost’s internal handling of imbalance

---

### 2. Models Used

* Logistic Regression (baseline model)
* XGBoost Classifier (final model)

---

### 3. Evaluation Metrics

Instead of accuracy, the following metrics were used:

* **Precision**
* **Recall**
* **F1 Score**

Reason:

> In fraud detection, missing a fraud (false negative) is more costly than a false alert.

---

##  Results

### Logistic Regression (at threshold = 0.8)

* Recall: ~0.88
* Precision: ~0.20
  
### RandomForest (at threshold = 0.3)
*Recall: 0.85
*Precision: 0.94

### XGBoost (at threshold = 0.2)
*Recall: 0.83
*Precision: 0.92 

---

##  Threshold Optimization

Instead of relying on the default threshold (0.5), probability thresholds were adjusted to improve fraud detection performance.

* Lower threshold → Higher recall (more fraud detected)
* Higher threshold → Higher precision (fewer false alarms)

Final model balances both aspects effectively.

---

##  Key Learnings

* Accuracy is not suitable for imbalanced datasets
* Precision–Recall tradeoff is critical in fraud detection
* Threshold tuning significantly impacts model performance
* XGBoost performs better than Logistic Regression on structured data

---

##  Future Improvements

* Implement Stratified K-Fold Cross Validation for more reliable evaluation
* Hyperparameter tuning using RandomizedSearchCV
* Cost-based optimization (real-world fraud vs false alert cost)
* Feature importance analysis and explainability

---

##  Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

##  Conclusion

The project demonstrates a practical approach to fraud detection by focusing on meaningful evaluation metrics and decision thresholds rather than default settings.
XGBoost combined with threshold tuning provides strong performance in identifying fraudulent transactions.

---

##  Acknowledgment

This project is built for learning and experimentation in real-world machine learning workflows, particularly handling imbalanced datasets and optimizing model decisions.

---
