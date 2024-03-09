from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):
    def __init__(self, X, y, is_train=True):
        """
        Custom dataset that applies normalization and SMOTE (for training set).
        :param X: Features
        :param y: Labels
        :param is_train: Flag to indicate if it is training data
        """
        if is_train:
            smote = SMOTE()
            X, y = smote.fit_resample(X, y)
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            # Apply normalization on validation data
            self.X = torch.from_numpy(X).float()
            self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx, is_train=True):
        return self.X[idx], self.y[idx]


def create_datasets(datapath, test_size=0.2):
    """
    Splits the data into train and validation datasets and applies preprocessing.
    :param X: Features
    :param y: Labels
    :param test_size: Size of the validation set
    """
    with open(datapath, 'rb') as handle:
        data = pd.read_pickle(handle)
    X_init = data['X_train']
    Y_init = data['Y']
    scaler = StandardScaler().fit(X_init)
    X_init = scaler.transform(X_init)
    X_train, X_val, y_train, y_val = train_test_split(
        X_init, Y_init, test_size=test_size, stratify=Y_init, random_state=42)
    train_dataset = CustomDataset(X_train, y_train, is_train=True)
    val_dataset = CustomDataset(X_val, y_val, is_train=False)
    return train_dataset, val_dataset
