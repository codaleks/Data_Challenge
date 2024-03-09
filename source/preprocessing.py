from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch


def balance_data(X, Y):
    sm = SMOTE(random_state=42)
    X_res, Y_res = sm.fit_resample(X, Y)
    return X_res, Y_res

def norm(data):
    data = (data - data.mean()) / data.std()
    return data

def preprocess_data(X, Y):
    X_norm = norm(X)
    X_train, X_val, Y_train, Y_val = train_test_split(X_norm, Y, test_size=0.2, random_state=42)
    X_res, Y_res = balance_data(X_train, Y_train)
    inputs = torch.tensor(X_res.values).float()
    labels = torch.tensor(Y_res.values).long()
    return inputs, labels, X_val, Y_val
