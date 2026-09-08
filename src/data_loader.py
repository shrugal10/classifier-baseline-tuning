import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y


def get_feature_names():
    dataset = load_breast_cancer(as_frame=True)
    return list(dataset.feature_names)


def get_target_names():
    dataset = load_breast_cancer(as_frame=True)
    return list(dataset.target_names)


if __name__ == "__main__":
    X, y = load_data()
    print(X.shape)
    print(y.value_counts())
