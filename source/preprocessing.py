import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from fairlearn.preprocessing import CorrelationRemover
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def open_data(datapath):
    with open(datapath, 'rb') as handle:
        data = pd.read_pickle(handle)
        X_init = data['X_train']
        Y_init = data['Y']
        S_init = data['S_train']
        columns = data['X_train'].columns
        S_init_reset = S_init.reset_index(drop=True)
    return X_init, Y_init, S_init_reset, columns


def split_data(X: pd.DataFrame, Y, S, test_size=0.2):
    X_concat = pd.concat([X, S], axis=1)
    # rename the last column to 'gender_class'
    X_concat.columns = X_concat.columns.astype(str)
    X_concat.columns.values[-1] = 'gender_class'

    X_train, X_val, y_train, y_val = train_test_split(
        X_concat, Y, test_size=test_size, stratify=Y, random_state=42)
    return X_train, X_val, y_train, y_val


def remove_correlation(X_train, X_val):
    X_S_train = pd.DataFrame(X_train)
    X_S_train.columns = X_S_train.columns.astype(str)

    X_S_val = pd.DataFrame(X_val)
    X_S_val.columns = X_S_val.columns.astype(str)

    cr = CorrelationRemover(sensitive_feature_ids=["gender_class"])
    X_cr_train = cr.fit_transform(X_S_train)
    X_cr_train = pd.DataFrame(X_cr_train)
    X_cr_train.set_index(X_S_train.index, inplace=True)
    X_cr_train.loc[:, "gender_class"] = X_S_train.loc[:, "gender_class"]
    X_cr_val = cr.transform(X_S_val)
    X_cr_val = pd.DataFrame(X_cr_val)
    X_cr_val.set_index(X_S_val.index, inplace=True)
    X_cr_val.loc[:, "gender_class"] = X_S_val.loc[:, "gender_class"]
    S_train_cr = X_cr_train['gender_class']
    S_val_cr = X_cr_val['gender_class']

    X_cr_train.drop(columns='gender_class', inplace=True)
    X_cr_val.drop(columns='gender_class', inplace=True)
    return X_cr_train, X_cr_val, S_train_cr, S_val_cr


def balance_data(X, y, S):
    S = S.to_numpy().reshape(-1, 1)
    X = np.concatenate([X, S], axis=1)
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
    S = X[:, -1]
    X = X[:, :-1]
    return X, y, S


def norm(data, columns):
    scaler = StandardScaler().fit(data)
    X_pre = scaler.transform(data)
    X_pre = pd.DataFrame(X_pre, columns=columns)
    X_pre_reset = X_pre.reset_index(drop=True)
    return X_pre_reset, scaler


def preprocess_data(datapath, test_size=0.2):
    X_init, Y_init, S_init, columns = open_data(datapath)
    X_pre, scaler = norm(X_init, columns)

    X_train, X_val, y_train, y_val = split_data(
        X_pre, Y_init, S_init, test_size)

    X_cr_train, X_cr_val, S_train_cr, S_val_cr = remove_correlation(
        X_train, X_val)

    X_train, y_train, S_train = balance_data(X_cr_train, y_train, S_train_cr)

    return X_train, X_cr_val, y_train, y_val, S_train, S_val_cr, scaler
