import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    X = pd.read_csv("train.feats.csv", dtype={'אבחנה-Surgery name3': 'str', "אבחנה-Ivi -Lymphovascular invasion": 'str',
                                              'אבחנה-Surgery date3': 'str'})
    y = pd.read_csv("train.labels.0.csv")
    train_X, train_y, test_X, test_y = train_test_split(X.to_numpy(), y.to_numpy(), train_size=10000)


if __name__ == '__main__':
    # np.random.seed(0)
    load_data()

    # Question 1 - Load and preprocessing of city temperature dataset
