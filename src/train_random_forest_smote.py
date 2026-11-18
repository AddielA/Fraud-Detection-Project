import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from data_loader import load_data, Root
from preprocess import engineer_features, split_and_scale
from evaluate import print_report

def main():

    df = engineer_features(load_data())
    X_train, X_test, y_train, y_test, _ = split_and_scale(df)

    print("\n Random Forest")
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f" Training set after SMOTE: {len(X_train_res):,} samples "
          f"({y_train_res.sum():,} fraud, {(y_train_res == 0).sum():,} normal)")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train_res, y_train_res)

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    print("\n Top 10 Features:")
    print("=" * 50)
    print(importances.sort_values(ascending=False).head(10))
    print("=" * 50)
    print("\n")

    # Metrics
    print_report(rf, X_test, y_test)
    
    
    joblib.dump(rf, Root / 'models' / 'random_forest_smote.pkl')

if __name__ == '__main__':
    main()