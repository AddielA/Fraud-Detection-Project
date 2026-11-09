import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from data_loader import load_data
from preprocess import engineer_features, split_and_scale

def main(): 
    df = load_data()
    print("\n Cost Information:")
    print(f" Average transaction amoun: ${df['Amount'].mean():.2f}")
    print(f" Average fraud amount:      ${df[df['Class'] == 1]['Amount'].mean():.2f}")
    
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(df)
    
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print("\n Data splits:")
    print(f" Training:   {len(X_train_fit):,} samples ({y_train_fit.sum()} fraud)")
    print(f" Validation: {len(X_val):,} samples ({y_val.sum()} fraud)")
    print(f" Test:       {len(X_test):,} samples ({y_test.sum()} fraud)")
    
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: weights[0], 1: weights[1]}
    
    print("\n Class Weights (for balanced learning):")
    print(f" Normal transactions (0): {weights[0]:.2f}")
    print(f" Fraud transactions  (1): {weights[1]:.2f}")
    print(f" Weight ratio:            {weights[1] / weights[0]:.0f}:1")
    
if __name__ == '__main__':
    main()