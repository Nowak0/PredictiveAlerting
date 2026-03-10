import numpy as np
import pandas as pd


TRAIN_PERCENTAGE = 0.8


def load_data():
    metrics = pd.read_parquet("dataset/20.parquet")
    feature_columns = ['total_power_avg', 'gpu0_core_temp_avg', 'gpu0_mem_temp_avg', 'ps0_output_curre_avg', 'ps0_input_voltag_avg']
    # feature_columns = metrics.select_dtypes(include=[np.number]).columns

    X = metrics[feature_columns].copy()
    y = metrics['value']
    X = X.fillna(0)
    y = y.fillna(0)

    return X, y


def create_samples(X, y, past_steps=5, future_steps=3):
    X_batches = []
    y_batches = []
    for i in range(past_steps, X.shape[0] - future_steps + 1):
        X_batches.append(X.iloc[i-past_steps:i].values.flatten())
        y_batch = y.iloc[i:i+future_steps].values
        y_batches.append(1 if y_batch.sum() > 0 else 0)

    X_batches = np.array(X_batches)
    y_batches = np.array(y_batches)
    train_sample_length = int(TRAIN_PERCENTAGE * X_batches.shape[0])

    X_train = X_batches[:train_sample_length]
    y_train = y_batches[:train_sample_length]
    X_test = X_batches[train_sample_length:]
    y_test = y_batches[train_sample_length:]

    return (X_train, y_train), (X_test, y_test)
