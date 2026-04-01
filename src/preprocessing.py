import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop("ID", axis=1)
    return df

def split_data(df):
    X = df.drop('default.payment.next.month', axis=1)
    y = df['default.payment.next.month']
    
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )