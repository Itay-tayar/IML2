import matplotlib
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names_without_zeros = ['floors', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'condition',
                       'grade', 'sqft_living15', 'sqft_lot15']
names_with_zeros = ['waterfront', 'view']


def fit_linear_regression(X, y):
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    sd = np.linalg.pinv(s)
    Xd = vh.T @ sd @ u.T
    w_hat = Xd.T.dot(y)
    return w_hat, s


def predict(X, w):
    return X.T @ w


def mse(y, yh):
    return ((yh - y)**2).mean(axis=None)
    # return np.average((yh-y)**2)


def load_data(path, shuffle=False):
    data = pd.read_csv(path)
    data = data.dropna()
    data.drop(data[data['price'] <= 0].index, inplace=True)
    for name in names_without_zeros:
        data.drop(data[data[name] <= 0].index, inplace=True)
    for name in names_with_zeros:
        data.drop(data[data[name] < 0].index, inplace=True)
    data['date'] = data['date'].map(lambda x: 0 if str(x).startswith("2014") is True else 1)
    y = (data['price'])
    data = data.drop(columns=['price', 'id'])
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    return data, y


def plot_singular_values(singulars):
    singulars = sorted(singulars, reverse=True)
    plt.figure()
    sing_vals = np.arange(len(singulars)) + 1
    plt.plot(sing_vals, singulars, 'ro-', linewidth=2)
    plt.title('Scree Plot - Singular Values')
    plt.ylabel('Singular Values')
    plt.show()


def question15():
    data, y = load_data("kc_house_data.csv")
    w_hat, s = fit_linear_regression(data.T, y)
    plot_singular_values(s)


def question16():
    data, y = load_data("kc_house_data.csv", True)
    data = np.array(data)
    y = np.array(y)
    split_index = int(0.75 * len(data))
    train_set = data[:split_index]
    y_train = y[:split_index]
    test_set = data[split_index:]
    y_test = y[split_index:]

    def mse_for_p(p):
        split_index_for_p = int(p/100 * len(train_set))
        train_set_p = train_set[:split_index_for_p]
        y_train_p = y_train[:split_index_for_p]
        w_hat, s = fit_linear_regression(train_set_p.T, y_train_p)
        y_hat = predict(test_set.T, w_hat)
        return mse(y_test, y_hat)

    # X = X_train_set[:int(np.round((p / 100) * len(X_train_set)))]

    mse_for_all_p = np.vectorize(mse_for_p)
    p_array = np.arange(1, 101)
    all_mse = mse_for_all_p(p_array)
    plt.figure()
    plt.plot(p_array, all_mse, 'ro-', linewidth=2)
    plt.title('MSE as function of p%')
    plt.ylabel('MSE')
    plt.show()


def feature_evaluation(X, y):
    for name in names_without_zeros:
        plt.figure()
        plt.plot(X[name], y, 'ro')
        pearson_cor = np.cov(X[name], y) / (np.sqrt(np.var(X[name])) * np.sqrt(np.var(y)))
        plt.title(name + ' as function of price\npearson correlation :\n ' + str(pearson_cor))
        plt.ylabel('price')
        plt.xlabel(name)
        plt.show()



data, y = load_data("kc_house_data.csv")
feature_evaluation(data,y)
