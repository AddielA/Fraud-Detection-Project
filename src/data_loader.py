import pandas as pd

def load_data(file_path='../fraud_anomaly_detection/data/raw/creditcard.csv'):
    df = pd.read_csv(file_path)

    print(" Dataset Overview ")
    print("=" * 50)
    print(f" Shape: {df.shape[0]:,} transactions x {df.shape[1]} features")
    print(f" Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f" Time span: {df['Time'].max() / 3600:.1f} hours")
    print(f"\n Column names:")
    print(df.columns.tolist()) 

    return df