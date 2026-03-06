import numpy as np
import pandas as pd


TRAIN_PERCENTAGE = 0.8

def load_data():
    metrics = pd.read_csv("dataset/4768977_timeseries_data.csv")
    bool_columns = metrics.select_dtypes(include=['bool']).columns
    metrics[bool_columns] = metrics[bool_columns].astype(int)
    feature_columns = metrics.select_dtypes(include=['int64', 'float64']).columns

    X = metrics[feature_columns].copy()
    X = X.fillna(0)
    y = metrics['alert_summary_status_general'].map({'TP': 1, 'SP': 1, 'TN': 0})

    return X, y

def create_samples(X, y, past_steps=12, future_steps=5):
    X_windows = []
    y_windows = []
    for i in range(past_steps, X.shape[0] - future_steps + 1):
        X_windows.append(X.iloc[i-past_steps:i].values.flatten())
        y_window = y.iloc[i:i+future_steps].values
        y_windows.append(1 if y_window.sum() > 0 else 0)

    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)

    train_sample_length = int(TRAIN_PERCENTAGE * X_windows.shape[0])
    X_train = X_windows[:train_sample_length]
    y_train = y_windows[:train_sample_length]
    X_test = X_windows[train_sample_length:]
    y_test = y_windows[train_sample_length:]
    return (X_train, y_train), (X_test, y_test)

