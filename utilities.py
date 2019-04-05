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


def init_parameters(K, d, mean, std):
    W = np.random.normal(loc=mean, scale=std, size=(K, d))
    b = np.random.normal(loc=mean, scale=std, size=(K, 1))
    return W, b


def normalize(X):
    mean = np.mean(X, axis=1, keepdims=True)
    X_normalized = X - mean
    return X_normalized, mean


def softmax(x):
    a = np.exp(x)
    return a / np.sum(a, axis=0)


def shuffleTrain(data):
    N = data['X_train'].shape[1]
    indeces = np.arange(N)
    np.random.shuffle(indeces)
    data['X_train'][:] = data['X_train'][:, indeces]


# def minibatchGD(data, GDparams, W, b, lambda_L2, verbose):
#     N = data['X_train'].shape[1]
#     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': []}
#
#     n_batches = N // GDparams['batch_size']
#
#     for epoch in range(GDparams['n_epochs']):
#         shuffleTrain(data)
#
#         update_history(history, data, W, b, lambda_L2, epoch, verbose)
#
#         for j in range(n_batches):
#             X_batch, Y_batch = get_mini_batch(data['X_train'], data['Y_train'], GDparams['batch_size'], j)
#
#             print()
#
#     return W, b, history
