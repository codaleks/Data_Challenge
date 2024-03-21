import torch

from source.preprocessing import preprocess_data
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y, S, is_train=True):
        self.is_train = is_train
        if is_train:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
            self.S = torch.from_numpy(S).long()
        else:
            self.X = torch.tensor(X.values, dtype=torch.float32)
            self.y = torch.tensor(y.values, dtype=torch.long)
            self.S = torch.tensor(S.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.S[idx]


def create_datasets(datapath, test_size=0.2):
    X_train, X_val, y_train, y_val, S_train, S_val, scaler = preprocess_data(
        datapath, test_size=0.2)

    train_dataset = CustomDataset(
        X_train, y_train, S_train, is_train=True)

    val_dataset = CustomDataset(
        X_val, y_val, S_val, is_train=False)

    return train_dataset, val_dataset, scaler
