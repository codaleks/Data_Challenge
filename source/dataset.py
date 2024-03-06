from torch.utils.data import DataLoader, Dataset
import pickle
import sys
import os
import preprocessing


class ChallengeDataset(Dataset):
    def __init__(self):
        sys.path.append(os.path.abspath(os.path.join('..', 'source')))
        with open('../data/data-challenge-student.pickle', 'rb') as handle:
            data = pickle.load(handle)
        self.X = data['X_train']
        self.Y = data['Y']
        self.S = data['S_train']

    def __len__(self):
        return len(self.X)

    def __getitem__(self):
        X_train, X_test, Y_train, Y_test, S_train, S_test = preprocessing(
            self.X, self.Y, self.S)
        return X_train, X_test, Y_train, Y_test, S_train, S_test
