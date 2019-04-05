from utilities import readDataset, normalize
from Model import Model

if __name__ == '__main__':
    # Loading datasets
    X_train, Y_train, y_train = readDataset('Datasets/cifar-10-batches-py/data_batch_1')
    X_val, Y_val, y_val = readDataset('Datasets/cifar-10-batches-py/data_batch_2')
    X_test, Y_test, y_test = readDataset('Datasets/cifar-10-batches-py/test_batch')

    # Data characteristics
    d, N = X_train.shape
    K = y_train.max() + 1

    # Gradient Descent parameters
    GDparams = {'batch_size': 100,
                'eta': 0.001,
                'n_epochs': 40,
                'lambda_L2': 0.0}

    # Normalization of the dataset
    X_train, mean = normalize(X_train)
    X_val = X_val - mean
    X_test = X_test - mean

    model = Model()
    model.compile(d, K, m=50)

    model.fit((X_train, Y_train, y_train), GDparams, val_data=(X_val, Y_val, y_val))

    print()
