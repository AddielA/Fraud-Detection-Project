# Overview #

## Credit Card Fraud Detection ##

A machine learning pipeline that detects fraudulent credit card transactions in a
dataset where only 0.172% of 284,807 transactions are fraud. It compares 10 models
across 6 algorithms, from Logistic Regression to XGBoost, handling the extreme
class imbalance with class weighting and SMOTE.

The best model, XGBoost, caught 82 of 98 frauds in the held-out test set with only
9 false alarms among 56,962 transactions (PR-AUC 0.88). Models are ranked by PR-AUC
rather than accuracy, since a model that flags nothing would still be 99.8% accurate.