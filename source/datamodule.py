from torch.utils.data import random_split,  DataLoader
import pytorch_lightning as pl
from typing import Optional
from preprocessing import preprocessing


class DataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def prepare_data(self):
        X_train, X_val, Y_train, Y_val, S_train, S_val = preprocessing()
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.S_train = S_train
        self.S_val = S_val

    def setup(self):
        Y_train = Y_train.to_numpy()
        Y_test = Y_test.to_numpy()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)
