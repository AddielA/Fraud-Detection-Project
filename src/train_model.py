import joblib
from sklearn.linear_model import LogisticRegression

from data_loader import load_data
from preprocess import engineer_features, split_and_scale
from evaluate import evaluate_model

def main():
    df = engineer_features(load_data())
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)

    metrics = evaluate_model(lr, X_test, y_test)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    joblib.dump(lr, 'models/logistic_regression.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

if __name__ == '__main__':
    main()