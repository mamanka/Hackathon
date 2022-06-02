import math

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

her2_title = 'Her2'
her2_labels = [0,1,2,3]
basic_stage_title = 'אבחנה-Basic stage'

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
    # test_y['אבחנה-Location of distal metastases'] = test_y['אבחנה-Location of distal metastases'].tolist()
    # train_y['אבחנה-Location of distal metastases'] = train_y['אבחנה-Location of distal metastases'].tolist()
    train_x = train_x.drop(["אבחנה-Tumor depth", "אבחנה-Surgery name3", "אבחנה-Surgery name2", "אבחנה-Tumor width",
                            "אבחנה-Surgery date3", "אבחנה-Surgery date2", "אבחנה-Ivi -Lymphovascular invasion"],
                           axis=1)
    test_x = test_x.drop(["אבחנה-Tumor depth", "אבחנה-Surgery name3", "אבחנה-Surgery name2", "אבחנה-Tumor width",
                          "אבחנה-Surgery date3", "אבחנה-Surgery date2", "אבחנה-Ivi -Lymphovascular invasion"],
                         axis=1)


    train_x.rename(columns={'אבחנה-Her2': 'Her2', 'אבחנה-Age': 'Age','אבחנה-T -Tumor mark (TNM)':'Tumor mark'
                            ,'אבחנה-Basic stage':'Basic stage'}, inplace=True)




    train_x[her2_title] = train_x['Her2'].fillna(0)
    train_x.replace({her2_title: {'neg': 1, 'NEG': 1, 'Neg': 1, 'negative': 1,
                                  'Neg ( FISH non amplified)': 1, 'NEGATIVE PER FISH': 1,
                                  'negative by FISH': 1, 'NEGATIVE': 1, 'Negative': 1,
                                  'Neg by IHC and FISH': 1, 'Neg by FISH': 1,
                                  'Negative ( FISH 1.0)': 1, '0': 1, 'neg.': 1}}, inplace=True)
    train_x.replace({her2_title: {'-': 0, '(-)': 0}}, inplace=True)
    train_x.replace({her2_title: {'+2 IHC': 2, '2+': 2,
                                  'Neg vs +2': 2, '+2 Fish NEG': 2, '+2 FISH-neg': 2,
                                  '+2 FISH negative': 2}}, inplace=True)
    train_x.replace({her2_title: {'FISH pos': 3, 'Positive by FISH': 3, 'pos': 3,
                                  '+3 100%cells': 3, '+3 100%': 3, 'Pos by FISH': 3, 'positive': 3,
                                  'FISH POS': 3, '+2 FISH-pos': 3, '+2 FISH(-)': 3,
                                  '+2, FISH חיובי': 3, 'Pos. FISH=2.9': 3, '+3 (100%cells)': 3,
                                  '+2 FISH positive': 3, 'חיובי': 3}}, inplace=True)
    train_x[her2_title] = train_x[her2_title].apply(lambda x: x if x in her2_labels else 0)

    new_train_x = train_x[["Age", "Her2",'Tumor mark','Basic stage']]

    DECIDED_AGE = 40

    # ages = pd.DataFrame(new_train_x["Age"])
    new_train_x["Age"] = np.where(new_train_x["Age"].to_numpy() < 40, 0,1)
    # for index, row in ages.iterrows():
    #     if row < DECIDED_AGE:
    #         row = 0
    #     else:
    #         row = 1

    new_train_x = pd.get_dummies(new_train_x, prefix='Tumor mark', columns=['Tumor mark'])
    train_x = pd.get_dummies(train_x, prefix='אבחנה-Side', columns=['אבחנה-Side'])





if __name__ == '__main__':
    # np.random.seed(0)
    load_data()

    # Question 1 - Load and preprocessing of city temperature dataset
