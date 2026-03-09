from handling_data import load_data, create_samples
from random_forest import run_advanced_random_forest, run_basic_random_forest


def main():
    X, y = load_data()
    train_data, test_data = create_samples(X, y)
    accuracy1, recall1, precision1 = run_basic_random_forest(*train_data, *test_data)
    print(accuracy1, recall1, precision1)
    # accuracy2, recall2, precision2 = run_advanced_random_forest(*train_data, *test_data, X)
    # print(accuracy2, recall2, precision2)


if __name__ == "__main__":
    main()
