from handling_data import load_data, create_samples
from model_random_forest import RandomForest
from model_xgboost import XGBoost


def alternative_model(train_data, test_data):
    rf = RandomForest()
    rf.train(*train_data)
    return rf.evaluate(*test_data)


def preferred_model(train_data, test_data):
    xgb = XGBoost()
    xgb.train(*train_data)
    return xgb.evaluate(*test_data)


def main():
    X, y = load_data()
    train_data, test_data = create_samples(X, y)

    accuracy, recall, precision = preferred_model(train_data, test_data)
    print(accuracy, recall, precision)

    accuracy, recall, precision = alternative_model(train_data, test_data)
    print(accuracy, recall, precision)


if __name__ == "__main__":
    main()
