import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_linear_regression(X, y):
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    sd = np.linalg.pinv(np.diag(s))
    Xd = vh.T @ sd @ u.T
    w_hat = Xd.T.dot(y)
    return w_hat, s


def question18():
    return pd.read_csv("covid19_israel.csv")


def question19():
    data = question18()
    data['log_detected'] = np.log(data['detected'])
    return data


def question20():
    data = question19()
    return fit_linear_regression(np.array(data['day_num']).reshape((1, 38)), data['log_detected'])


def question21():
    data = question19()
    w, s = question20()
    plt.figure()
    plt.plot(data['day_num'], data['log_detected'], 'ro', linewidth=2, label='log_detected')
    plt.plot(data['day_num'], np.array(data['day_num']).reshape((1, 38)).T @ w, 'b-', linewidth=2, label='estimated')
    plt.title('log_detected as function of day_num')
    plt.ylabel('log_detected')
    plt.xlabel('day_num')
    plt.legend(loc="upper left")
    plt.figure()
    plt.plot(data['day_num'], data['detected'], 'ro', linewidth=2, label='detected')
    plt.plot(data['day_num'], np.exp(np.array(data['day_num']).reshape((1, 38)).T @ w), 'b-', linewidth=2,
             label='estimated')
    plt.title('detected as function of day_num')
    plt.ylabel('detected')
    plt.xlabel('day_num')
    plt.legend(loc="upper left")
    plt.show()
