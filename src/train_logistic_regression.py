import joblib
from sklearn.linear_model import LogisticRegression

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
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced' 
    )
    lr.fit(X_train, y_train)

    # 3. Evaluate
    print_report(lr, X_test, y_test)

    # 4. Save
    joblib.dump(lr, Root / 'models' / 'logistic_regression.pkl')
    joblib.dump(scaler, Root / 'models' / 'scaler.pkl')


if __name__ == '__main__':
    main()