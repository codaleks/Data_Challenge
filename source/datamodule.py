import pytorch_lightning as pl
import torch
from source.dataset import ChallengeDataset
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 32, data_path="A:\MSBGD\Data_Challenge\data\data-challenge-student.pickle"):
        self.batch_size = batch_size
        self.data_path = data_path

    def prepare_data_per_node(self):
        pass

    def setup(self, stage=None):
        self.prepare_data()
        dataset = ChallengeDataset(data_path=self.data_path)
        n_train = int(len(dataset) * 0.7)
        n_val = int(len(dataset) * 0.2)
        n_test = len(dataset) - n_train - n_val
        self.train, self.val, self.test = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test])

    def train_dataloader(self):
        # concatenate X_train and Y_train
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=8)
