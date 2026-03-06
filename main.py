from handling_data import load_data, create_samples
from random_forest import run_random_forest


def main():
    X, y = load_data()
    train_data, test_data = create_samples(X, y)
    run_random_forest(*train_data, *test_data)


if __name__ == "__main__":
    main()
