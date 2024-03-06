import matplotlib.pyplot as plt
import pickle
import sys
import os

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def balance_data(X, Y, S):
    sm = SMOTE(random_state=42)
    X['S'] = S
    X.columns = X.columns.astype(str)
    X_res, Y_res = sm.fit_resample(X, Y)
    S_res = X_res['S']
    X_res = X_res.drop(columns=['S'])
    return X_res, Y_res, S_res


def visualize_pca(data):
    pca = PCA()
    pca.fit(data)
    # cumulative explained variance
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
    plt.plot(cumulative_explained_variance)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title(
        'cumulative explained variance as a function of the number of components')
    plt.show()


def pca(data, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    X_pca = pca.transform(data)
    print("X_pca shape :", X_pca.shape)
    return X_pca


def visualize_hist(data):
    plt.hist(data)
    plt.title('Distribution of S')
    plt.show()


def preprocessing(X, Y, S):
    # Balance data
    X_res, Y_res, S_res = balance_data(X, Y, S)

    # Visualize distribution of data
    # visualize_hist(S_res)

    # Visualize PCA
    # visualize_pca(X_res)

    # PCA
    X_pca = pca(X_res)
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        X_pca, Y_res, S_res, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test, S_train, S_test
