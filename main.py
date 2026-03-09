from handling_data import load_data, create_samples
from models import run_random_forest, run_xgboost


def main():
    X, y = load_data()
    train_data, test_data = create_samples(X, y)

    accuracy1, recall1, precision1 = run_random_forest(*train_data, *test_data)
    print(accuracy1, recall1, precision1)

    accuracy2, recall2, precision2 = run_xgboost(*train_data, *test_data)
    print(accuracy2, recall2, precision2)


if __name__ == "__main__":
    main()
