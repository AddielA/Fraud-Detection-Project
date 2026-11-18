import joblib
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from data_loader import load_data, Root
from preprocess import engineer_features, split_and_scale
from evaluate import print_report
import warnings
warnings.filterwarnings('ignore')


def main():
    # 1. Load and prepare data
    df = engineer_features(load_data())
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
        
    # 2. Train
    print(" Training Logistic Regression...")
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f" Training set after SMOTE: {len(X_train_res):,} samples "
          f"({y_train_res.sum():,} fraud, {(y_train_res == 0).sum():,} normal)")
    
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced' 
    )
    lr.fit(X_train_res, y_train_res)

    # 3. Evaluate
    print_report(lr, X_test, y_test)

    # 4. Save
    joblib.dump(lr, Root / 'models' / 'logistic_regression_smote.pkl')
    joblib.dump(scaler, Root / 'models' / 'scaler.pkl')


if __name__ == '__main__':
    main()