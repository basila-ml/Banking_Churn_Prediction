# Bank Churn Prediction

A machine learning project that predicts customer churn for a bank's credit card
portfolio, using exploratory data analysis, SMOTE class balancing, PCA, and a
comparison of multiple classification models.

## Overview

Customer churn — when a customer closes their account or stops using their credit
card — is one of the costliest problems a bank can face, since acquiring a new
customer is far more expensive than retaining an existing one. This project analyzes
real credit-card customer records to understand why customers leave and builds a
model that predicts churn before it happens.

## Dataset

~10,000 bank customers with features covering:
- **Demographics:** age, gender, income, education, marital status
- **Account details:** card category, credit limit, months on book
- **Behavior:** transaction amount/count, relationship count, inactivity

Target variable: `Attrition_Flag` (churned vs. existing customer).

Place `BankChurners.csv` in the project root before running the notebook.

## Approach

1. **Exploratory Data Analysis** — distributions and breakdowns across age, gender,
   income, education, marital status, card category, and transaction behavior.
2. **Preprocessing** — label encoding, one-hot encoding of categorical features,
   removal of identifier columns.
3. **Train/test split** — stratified split on the target, done before any resampling.
4. **Class balancing** — SMOTE applied to the training set only, to address the
   ~16% churn rate.
5. **Dimensionality reduction** — PCA on the one-hot encoded categorical features.
6. **Modeling** — K-Nearest Neighbors, Support Vector Classifier, Random Forest,
   Logistic Regression, and a soft-voting ensemble of all four.
7. **Evaluation** — accuracy, classification report, confusion matrix, and ROC curve
   for each model, plus a final side-by-side accuracy comparison.

## Requirements

- Python 3
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
``
## Results

| Model | Accuracy |
|---|---|
| KNN | 0.858 |
| Random Forest | 0.855 |
| Voting Classifier | 0.854 |
| SVC | 0.850 |
| Logistic Regression | 0.814 |

