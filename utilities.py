import numpy as np


def readDataset(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    X = dict[b'data'].T
    y = np.array(dict[b'labels'])
    Y = make_one_hot(y).T
    return X, Y, y


def make_one_hot(x):
    one_hot_x = np.zeros((x.size, x.max() + 1))
    one_hot_x[np.arange(x.size), x] = 1
    return one_hot_x


def normalize(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    X_normalized = (X - mean ) / std
    return X_normalized, mean, std


def softmax(x):
    a = np.exp(x)
    return a / np.sum(a, axis=0)
