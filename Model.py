import numpy as np
from tqdm import trange

from utilities import softmax


class Model:

    def __init__(self):
        # Data parameters
        self.d = None  # Features
        self.K = None  # Classes
        self.m = None  # Hidden units

        # Model hyperparameters
        self.lambda_L2 = None
        self.batch_size = None
        self.n_epochs = None

        # Model parameters
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # History of training
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def compile(self, d, K, m=50):
        self.d = d
        self.K = K
        self.m = m
        self.W1 = np.random.normal(loc=0.0, scale=1 / np.sqrt(d), size=(m, d))
        self.b1 = np.zeros((m, 1))
        self.W2 = np.random.normal(loc=0.0, scale=1e-3, size=(K, m))
        self.b2 = np.zeros((K, 1))

    def forward_pass(self, X):
        S1 = self.W1.dot(X) + self.b1
        H = np.maximum(0, S1)
        S = self.W2.dot(H) + self.b2
        P = softmax(S)
        return H, P

    def compute_cost(self, X, y, P=None):
        N = X.shape[1]
        if P is None:
            P = self.forward_pass(X)[1]

        loss = -np.log(P[y, np.arange(N)]).mean()
        reg_term = self.lambda_L2 * (np.square(self.W1).sum() + np.square(self.W2).sum())

        return loss + reg_term

    def compute_accuracy(self, X, y):
        P = self.forward_pass(X)[1]
        predictions = np.argmax(P, axis=0)
        accuracy = (predictions == y).mean()
        return accuracy

    def evaluate(self, X, y):
        cost = self.compute_cost(X, y)
        accuracy = self.compute_accuracy(X, y)
        return cost, accuracy

    def backward_pass(self, X, Y, H, P):
        ones_nb = np.ones(self.batch_size)
        c = 1 / self.batch_size
        G = P - Y
        dL_dW2 = c * G.dot(H.T)
        dL_db2 = c * G.dot(ones_nb).reshape(self.K, 1)
        G = self.W2.T.dot(G)
        G = G * (H > 0)
        dL_dW1 = c * G.dot(X.T)
        dL_db1 = c * G.dot(ones_nb).reshape(self.m, 1)

        return dL_dW1, dL_db1, dL_dW2, dL_db2

    def fit(self, train_data, GDparams, val_data):
        self._saveGDparams(GDparams)

        N = train_data[0].shape[1]

        self._run_epochs(train_data, val_data, N)

    def _run_epochs(self, train_data, val_data, N):
        n_batches = N // self.batch_size

        for_epoch = trange(self.n_epochs, leave=True)

        for epoch in for_epoch:
            self.shuffleData(train_data)

            # Evaluate for saving in history
            train_cost, train_acc = self.evaluate(train_data[0], train_data[2])
            val_cost, val_acc = self.evaluate(val_data[0], val_data[2])
            for_epoch.set_description(f'train_loss: {train_cost:.4f}\ttrain_acc: {100*train_acc:.2f}%'+ ' | '
                                      f'val_loss: {val_cost:.4f}\ttrain_acc: {100*val_acc:.2f}% ')

            self._run_batches(train_data, n_batches)

    def _run_batches(self, train_data, n_batches):

        for b in range(n_batches):
            X_batch, Y_batch, y_batch = self._get_mini_batch(train_data, b)

            self._update_weights(X_batch, Y_batch, y_batch)

    def _update_weights(self, X_batch, Y_batch, y_batch):
        H, P = self.forward_pass(X_batch)
        dL_dW1, dL_db1, dL_dW2, dL_db2 = self.backward_pass(X_batch, Y_batch, H, P)

        self.W1 -= self.eta * dL_dW1 + 2 * self.lambda_L2 * self.W1
        self.b1 -= self.eta * dL_db1
        self.W2 -= self.eta * dL_dW2 + 2 * self.lambda_L2 * self.W2
        self.b2 -= self.eta * dL_db2

    def _saveGDparams(self, GDparams):
        self.lambda_L2 = GDparams['lambda_L2']
        self.batch_size = GDparams['batch_size']
        self.n_epochs = GDparams['n_epochs']
        self.eta = GDparams['eta']

    def _get_mini_batch(self, data, j):
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size
        X_batch = data[0][:, j_start:j_end]
        Y_batch = data[1][:, j_start:j_end]
        y_batch = data[2][j_start:j_end]
        return X_batch, Y_batch, y_batch

    def shuffleData(self, data):
        """
        Shuffle a dataset made of X, Y and y.
        """
        N = data[0].shape[1]
        indeces = np.arange(N)
        np.random.shuffle(indeces)
        data[0][:] = data[0][:, indeces]
        data[1][:] = data[1][:, indeces]
        data[2][:] = data[2][indeces]


