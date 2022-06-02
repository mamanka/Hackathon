import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit


def load_data():
    X = pd.read_csv("train.feats.csv", dtype={'אבחנה-Surgery name3': 'str', "אבחנה-Ivi -Lymphovascular invasion": 'str',
                                              'אבחנה-Surgery date3': 'str'})
    y = pd.read_csv("train.labels.0.csv")

    # split data:
    splitter = GroupShuffleSplit(test_size=.80, n_splits=2, random_state=7)
    split = splitter.split(X, groups=X['User Name'])
    train_inds, test_inds = next(split)

    train_x = X.iloc[train_inds]
    test_x = X.iloc[test_inds]
    train_y = y.iloc[train_inds]
    test_y = y.iloc[test_inds]
    test_y['אבחנה-Location of distal metastases'] = test_y['אבחנה-Location of distal metastases'].tolist()
    train_y['אבחנה-Location of distal metastases'] = train_y['אבחנה-Location of distal metastases'].tolist()
    print(test_y, train_y)
    train_x.info()
    train_x.drop(["אבחנה-Tumor depth", "אבחנה-Surgery name3", "אבחנה-Surgery name2", "אבחנה-Tumor width",
                  "אבחנה-Surgery date3", "אבחנה-Surgery date2"],
                 axis=1, inplace=True)
    test_x.drop(["אבחנה-Tumor depth", "אבחנה-Surgery name3", "אבחנה-Surgery name2", "אבחנה-Tumor width",
                  "אבחנה-Surgery date3", "אבחנה-Surgery date2"],
                 axis=1, inplace=True)


if __name__ == '__main__':
    # np.random.seed(0)
    load_data()

    # Question 1 - Load and preprocessing of city temperature dataset
