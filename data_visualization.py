import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    X = pd.read_csv("train.feats.csv")
    y = pd.read.csv("train.labels.0.csv")
    j = 5
    train_X, train_y, test_X, test_y = train_test_split(pd.DataFrame(X), pd.Series(y), train_proportion=2 / 3)


if __name__ == '__main__':
    # np.random.seed(0)
    load_data()

    # Question 1 - Load and preprocessing of city temperature dataset
