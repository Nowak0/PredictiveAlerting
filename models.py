import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb


THRESHOLD = 0.6
N_ESTIMATORS = 300


def print_confusion_matrix(y_test, y_pred, title='Confusion matrix'):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(title)
    plt.show()

def run_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=N_ESTIMATORS)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > THRESHOLD).astype(int)

    print_confusion_matrix(y_test, y_pred, title='Confusion Matrix for Random Forest Model in Incident Detection')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, recall, precision

def run_xgboost(X_train, y_train, X_test, y_test):
    train_data = xgb.DMatrix(X_train, y_train)
    test_data = xgb.DMatrix(X_test, y_test)
    amount_no_incident = y_train[y_train == 0].shape[0]
    amount_incident = y_train[y_train == 1].shape[0]

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': amount_no_incident / amount_incident
    }
    model = xgb.train(params, train_data, num_boost_round=N_ESTIMATORS)

    y_prob = model.predict(test_data)
    y_pred = (y_prob > THRESHOLD).astype(int)

    print_confusion_matrix(y_test, y_pred, title='Confusion Matrix for XGBoost Model in Incident Detection')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, recall, precision
