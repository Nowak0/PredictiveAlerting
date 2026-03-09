import numpy as np
import pandas as pd

TRAIN_PERCENTAGE = 0.8

def load_data():
    metrics = pd.read_parquet("dataset/1/20.parquet")
    feature_columns = ['p0_power_avg', 'gpu0_core_temp_avg', 'gpu0_mem_temp_avg', 'fan0_0_avg']
    # feature_columns = ['gpu0_core_temp_avg', 'gpu0_mem_temp_avg']
    # feature_columns = metrics.select_dtypes(include=[np.number]).columns

    X = metrics[feature_columns].copy()
    X = X.fillna(0)
    y = metrics['value']
    y = y.fillna(0)

    return X, y

def create_samples(X, y, past_steps=5, future_steps=3):
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
