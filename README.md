# Credit Card Fraud Detection

An end-to-end machine learning pipeline for identifying fraudulent credit card transactions. The project walks through the full workflow — business framing, data exploration, feature engineering, model training, evaluation, and interpretability — using a highly imbalanced transaction dataset.

## Table of Contents

- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Dataset](#dataset)
- [Features Engineered](#features-engineered)
- [Modeling Approach](#modeling-approach)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Next Steps](#next-steps)

## Overview

Credit card fraud is rare compared to legitimate activity, which makes it a classic imbalanced classification problem. The goal of this project is to build a binary classifier that flags a transaction as:

- **0** — Legitimate transaction
- **1** — Fraudulent transaction

The system is designed to catch as many fraudulent transactions as possible (high recall) while keeping false alarms on legitimate customers to a reasonable level (acceptable precision).

## Project Workflow

The notebook (`Credit_Card_Fraud_Detection.ipynb`) follows a ten-stage pipeline:

1. **Business Understanding** — problem statement, business objective, ML objective, and success metrics
2. **Data Loading & Understanding** — load the CSV, inspect shape/dtypes, build a data dictionary
3. **Data Cleaning** — remove duplicates, check for missing/invalid values, inspect outliers via IQR
4. **Exploratory Data Analysis (EDA)** — class balance, amount distributions, correlations, fraud-by-hour patterns
5. **Feature Engineering** — derive time- and amount-based features
6. **Data Preprocessing** — train/test split, scaling & encoding pipeline, SMOTE resampling
7. **Model Training** — fit four candidate classifiers
8. **Model Evaluation** — precision, recall, F1, ROC-AUC, confusion matrices, precision-recall curves
9. **Feature Importance & Explainability** — Random Forest importances and permutation importance
10. **Conclusion & Next Steps** — summary of findings and recommended follow-up work

## Dataset

The notebook expects a CSV file at `/content/creditcard.csv` (update this path to match your environment) containing:

| Column | Description |
|---|---|
| `Time` | Seconds elapsed between this transaction and the first transaction in the dataset |
| `V1`–`V28` | PCA-transformed numerical features (anonymized for confidentiality) |
| `Amount` | Transaction amount |
| `Class` | Target variable — `0` = legitimate, `1` = fraudulent |

This matches the structure of the well-known [Kaggle "Credit Card Fraud Detection" dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

**Key characteristic:** fraudulent transactions make up a very small fraction of the total — the defining challenge tackled throughout the notebook.

## Features Engineered

Building on patterns found during EDA, the following features are added:

- `Hour` — hour of day derived from `Time`
- `Day_Night` — categorical flag (`Day` / `Night`) based on hour
- `Log_Amount` — log-transformed transaction amount (`log1p`)
- `Amount_Category` — binned amount (`Low`, `Medium`, `High`, `Very High`)
- `High_Value` — binary flag for transactions above the 95th percentile amount

## Modeling Approach

- **Preprocessing:** numerical features are standardized (`StandardScaler`); categorical features (`Day_Night`, `Amount_Category`) are one-hot encoded via a `ColumnTransformer`.
- **Class imbalance:** `SMOTE` (Synthetic Minority Over-sampling Technique) is applied to the **training set only**, generating synthetic fraud examples so models aren't biased toward the majority class. The test set is left untouched to reflect real-world class proportions.
- **Models trained:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

## Evaluation Metrics

Because fraud is rare, accuracy alone is misleading. The project instead prioritizes:

- **Recall** — proportion of actual fraud correctly detected (most important — missed fraud is costly)
- **Precision** — proportion of flagged transactions that are actually fraud
- **F1 Score** — harmonic mean of precision and recall
- **ROC-AUC** — overall ability to distinguish fraud from legitimate transactions

The best-performing model is selected based on **Recall**, in line with the business objective of catching as much fraud as possible.

## Results

Model comparison, confusion matrices, precision-recall curves, and feature importance charts (Random Forest importances and permutation importance) are generated in Sections 8 and 9 of the notebook. Run the notebook end-to-end to reproduce the exact metric values and plots for your data.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## How to Run

1. Download the dataset (e.g. from Kaggle) and place it locally, or update `file_path` in Section 2.1 of the notebook to point to your CSV.
2. Install the requirements listed above.
3. Open `Credit_Card_Fraud_Detection.ipynb` in Jupyter, JupyterLab, or Google Colab.
4. Run all cells in order — each section builds on the previous one (cleaning → EDA → feature engineering → preprocessing → training → evaluation → explainability).

## Project Structure

```
.
├── Credit_Card_Fraud_Detection.ipynb   # Main analysis & modeling notebook
└── README.md                           # This file
```


