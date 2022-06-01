import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    X = pd.read_csv("train.feats.csv")
    y = pd.read_csv("train.labels.0.csv")
    x_train, x_test, y_train, y_test = train_test_split(X.to_numpy(),y.to_numpy(),train_size=10000)
    j=5



if __name__ == '__main__':
    # np.random.seed(0)
    load_data()

    # Question 1 - Load and preprocessing of city temperature dataset
