import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
    }

def print_report(model, X_test, y_test):
    """Print a model's test-set metrics, fraud counts and classification report."""
    metrics = evaluate_model(model, X_test, y_test)
    print("\n Results:")
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\n Frauds caught: {tp}")
    print(f" Frauds missed: {fn}")
    print(f" False alarms: {fp}")
    print(f" Normal transactions correctly passed: {tn}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    return metrics

def evaluate_anomaly_model(model, X_test, y_test):
    y_pred = np.where(model.predict(X_test) == -1, 1, 0)   # -1 = anomaly -> 1 = fraud
    y_scores = -model.score_samples(X_test)                # negated: higher = more anomalous
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_scores),
        'pr_auc': average_precision_score(y_test, y_scores),
    }
