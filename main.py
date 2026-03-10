from handling_data import load_data, create_samples
from models import run_random_forest, run_xgboost


def alternative_model(train_data, test_data):
    return run_random_forest(*train_data, *test_data)

def preferred_model(train_data, test_data):
    return run_xgboost(*train_data, *test_data)

def main():
    X, y = load_data()
    train_data, test_data = create_samples(X, y)

    accuracy, recall, precision = preferred_model(train_data, test_data)
    print(accuracy, recall, precision)
    accuracy, recall, precision = alternative_model(train_data, test_data)
    print(accuracy, recall, precision)




if __name__ == "__main__":
    main()
