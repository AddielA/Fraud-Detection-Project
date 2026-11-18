import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_data, Root
from preprocess import engineer_features, split_and_scale
from evaluate import print_report

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

    # Metrics
    print_report(rf, X_test, y_test)
    joblib.dump(rf, Root / 'models' / 'random_forest.pkl')

if __name__ == '__main__':
    main()