import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_data
from preprocess import engineer_features, split_and_scale
from evaluate import evaluate_model

def main():

    df = engineer_features(load_data())
    X_train, X_test, y_train, y_test, _ = split_and_scale(df)

    print("\n Random Forest")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    print("\n Top 10 Features:")
    print("=" * 50)
    print(importances.sort_values(ascending=False).head(10))
    print("=" * 50)
    print("\n")

    # Predictions
    y_pred_rf = rf.predict(X_test)

    # Metrics
    print("\n Results:")
    print("=" * 50)
    metrics = evaluate_model(rf, X_test, y_test)
    for name, value in metrics.items():
        print(f" {name}: {value:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
    print(f"\n Frauds caught: {tp}")
    print(f" Frauds missed: {fn}")
    print(f" False alarms: {fp}")
    print(f" Normal transactions correctly passed: {tn}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    joblib.dump(rf, 'models/random_forest.pkl')

if __name__ == '__main__':
    main()