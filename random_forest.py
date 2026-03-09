from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd


THRESHOLD = 0.6


def print_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()

def run_basic_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > THRESHOLD).astype(int)

    print_confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, recall, precision


def run_advanced_random_forest(X_train, y_train, X_test, y_test, X):
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    params = {
        'n_estimators': [i for i in range(60,200,20)],
        'max_depth': [i for i in range(6,12)],
    }
    rand_search = RandomizedSearchCV(estimator=rf, param_distributions=params,
                                     n_iter=18, scoring='f1',
                                     n_jobs=-1, random_state=42, cv=5)
    rand_search.fit(X_train, y_train)

    best_rf = rand_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    print_confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, recall, precision
