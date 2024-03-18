from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from fairlearn.preprocessing import CorrelationRemover


class CustomDataset(Dataset):
    def __init__(self, X, y, S, is_train=True):
        self.is_train = is_train
        if is_train:
            S = S.to_numpy().reshape(-1, 1)
            X = np.concatenate([X, S], axis=1)
            smote = SMOTE()
            X, y = smote.fit_resample(X, y)
            S = X[:, -1]
            X = X[:, :-1]
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
            self.S = torch.from_numpy(S).long()
        else:
            self.X = torch.from_numpy(X).float()
            self.y = torch.tensor(y.values, dtype=torch.long)
            self.S = torch.tensor(S.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_train:
            return self.X[idx], self.y[idx], self.S[idx]
        else:
            return self.X[idx], self.y[idx], self.S[idx]


def create_datasets(datapath, test_size=0.2):
    with open(datapath, 'rb') as handle:
        data = pd.read_pickle(handle)
        X_init = data['X_train']
        Y_init = data['Y']
        S_init = data['S_train']

    scaler = StandardScaler().fit(X_init)
    X_pre = scaler.transform(X_init)

    X_pre = pd.DataFrame(X_pre, columns=data['X_train'].columns)
    X_concat = pd.concat([X_pre, S_init], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_concat, Y_init, test_size=0.2, stratify=Y_init, random_state=42)

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

    train_dataset = CustomDataset(
        X_cr_train, y_train, S_train_cr, is_train=True)
    val_dataset = CustomDataset(X_val, y_val, S_val_cr, is_train=False)

    return train_dataset, val_dataset, scaler
