import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
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


def pca_norm(data, n_components=0.95):
    pca = PCA(n_components=n_components)
    model = pca.fit(data)
    # save the pca model
    X_pca = pca.transform(data)
    # normalize the data
    X_pca = (X_pca - X_pca.mean()) / X_pca.std()
    return X_pca, model


def visualize_hist(data):
    plt.hist(data)
    plt.title('Distribution of S')
    plt.show()
