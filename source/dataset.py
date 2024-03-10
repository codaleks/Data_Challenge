from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


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
    X_init = scaler.transform(X_init)
    X_train, X_val, y_train, y_val, S_train, S_val = train_test_split(
        X_init, Y_init, S_init, test_size=test_size, stratify=Y_init, random_state=42)
    train_dataset = CustomDataset(X_train, y_train, S_train, is_train=True)
    val_dataset = CustomDataset(X_val, y_val, S_val, is_train=False)
    return train_dataset, val_dataset, scaler
