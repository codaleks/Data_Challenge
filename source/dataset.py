from torch.utils.data import Dataset
import torch
import pandas as pd
from torch.utils.data import random_split
from source.preprocessing import balance_data, pca_norm


class ChallengeDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as handle:
            data = pd.read_pickle(handle)
        self.X = data['X']
        self.Y = data['Y']
        self.S = data['S']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx], self.S[idx])


def prepare_data(data_path):
    dataset = ChallengeDataset(
        data_path=data_path)
    total_length = len(dataset)  # Total number of samples in the dataset

    # Define the proportions of the splits
    train_size = int(0.7 * total_length)  # 70% of the dataset for training
    valid_size = int(0.2 * total_length)  # 20% for validation
    # The rest for testing, to ensure no sample is left out
    test_size = total_length - train_size - valid_size
    # Perform the split
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size])
    return train_dataset, valid_dataset, test_dataset
