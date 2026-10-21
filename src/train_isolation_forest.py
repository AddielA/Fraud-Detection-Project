import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from data_loader import load_data
from preprocess import engineer_features, split_and_scale

def train_and_score(label, X_fit, X_test, y_test):

    """Fit an Isolation Forest on X_fit, score it on X_test, print and return metrics."""
    print("\n" + "=" * 60)
    print(f" {label}")
    print("=" * 60)
    print(f" Training on {len(X_fit):,} transactions")


    print("\n Random Forest")
    model = IsolationForest(
        contamination=0.002, 
        random_state=42,
        n_jobs=1
    )
    model.fit(X_fit)

    # Predictions
    y_pred = np.where(model.predict(X_test) == -1, 1, 0)
    
    # Get anomaly scores
    y_scores = -model.score_samples(X_test)
    
    # Metrics
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_pred),
        'pr_auc':    average_precision_score(y_test, y_pred)
    }
    
    print("\n Results:")
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")

    flagged = int(y_pred.sum())
    caught = int(((y_pred == 1) & (y_test == 1)).sum())
    total_fraud = int(y_test.sum())

    print(f"\n This shows us that the model flagged {flagged} transactions as "
          f"unusual, and {caught} of them were real fraud.")
    print(f" Based on recall, it caught {metrics['recall'] * 100:.1f}% of all "
          f"{total_fraud} frauds.")
    print(f" Based on precision, {metrics['precision'] * 100:.1f}% of its alerts "
          f"were actually fraud.")
    
    return model, metrics


def main():
    
    df = engineer_features(load_data())
    X_train, X_test, y_train, y_test, _ = split_and_scale(df)
    
    # Method 1: Fit all trinaing data
    model_all, metrics_all = train_and_score(
        "Method 1: Fit all training data",
        X_train, X_test, y_test
    )
    
    # Method 2: Fit on normal transactions only
    X_train_normal = X_train[y_train == 0]
    model_normal, metrics_normal = train_and_score(
        "Method 2: Fit on normal transactions only",
        X_train_normal, X_test, y_test
    )
    
    # Side-by-side comparison
    print("\n" + "=" * 60)
    print(" SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    comparison = pd.DataFrame(
        {'All data': metrics_all, 'Normal only': metrics_normal}
    )
    comparison['difference'] = comparison['Normal only'] - comparison['All data']
    print(comparison.round(4).to_string())

    pr_diff = comparison.loc['pr_auc', 'difference']
    print(f"\n PR-AUC changed by {pr_diff:+.4f} when training on normal transactions only.")
    print(" Fraud is only ~0.17% of the data, so removing it from training changes "
          "very little. Small differences may just be noise from one train/test split.")

    joblib.dump(model_all, 'models/isolation_forest_all.pkl')
    joblib.dump(model_normal, 'models/isolation_forest_normal.pkl')
    
if __name__ == '__main__':
    main()