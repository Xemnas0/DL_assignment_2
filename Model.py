import numpy as np
from tqdm import trange
from scipy import signal

from utilities import softmax

import matplotlib.pyplot as plt


class Model:

    def __init__(self):
        # Data parameters
        self.d = None  # Features
        self.K = None  # Classes
        self.m = None  # Hidden units

        # Gradient Descent hyperparameters
        self.lambda_L2 = None
        self.batch_size = None
        self.n_cycles = None
        self.n_epochs = None
        self.eta_min = None
        self.eta_max = None
        self.eta = None

        self.verbose = True
        self.scheduled_eta = None

        # Flag for checking the gradient numerically
        self.check_gradient = False

        # Model parameters
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # History of training
        self.history = {'train_cost': [], 'train_loss': [], 'train_acc': [],
                        'val_cost': [], 'val_loss': [], 'val_acc': [], 'eta': []}

    def compile(self, d, K, m=50):
        self.d = d
        self.K = K
        self.m = m
        self.W1 = np.random.normal(loc=0.0, scale=1 / np.sqrt(d), size=(m, d))
        self.b1 = np.zeros((m, 1))
        self.W2 = np.random.normal(loc=0.0, scale=1e-3, size=(K, m))
        self.b2 = np.zeros((K, 1))

    def _saveGDparams(self, GDparams):
        self.lambda_L2 = GDparams['lambda_L2']
        self.batch_size = GDparams['batch_size']
        self.n_cycles = GDparams['n_cycles']
        self.n_s = GDparams['n_s']
        self.eta_min = GDparams['eta_min']
        self.eta_max = GDparams['eta_max']
        self.verbose = GDparams['verbose']

        # Prepare schedule of the learning rate
        t = np.arange(self.n_s * 2)
        freq = 1 / (2 * self.n_s)
        self.scheduled_eta = (signal.sawtooth(2 * np.pi * t * freq, 0.5) + 1) / 2 * (
                self.eta_max - self.eta_min) + self.eta_min

        # Debug
        # plt.plot(self.scheduled_eta)
        # plt.show()

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

        return loss + reg_term, loss

    def compute_accuracy(self, X, y):
        P = self.forward_pass(X)[1]
        predictions = np.argmax(P, axis=0)
        accuracy = (predictions == y).mean()
        return accuracy

    def evaluate(self, X, y):
        """
        Computes cost and accuracy of the given data.
        """
        cost, loss = self.compute_cost(X, y)
        accuracy = self.compute_accuracy(X, y)
        return cost, loss, accuracy

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

    def fit(self, train_data, GDparams, val_data=None, val_split=None):
        assert val_data is not None or val_split is not None, 'Validation set not defined.'
        self._saveGDparams(GDparams)
        if val_data is None:
            train_data, val_data = self.split_data(train_data, val_split)

        N = train_data[0].shape[1]

        t_tot = self.n_cycles * 2 * self.n_s
        self.n_epochs = int(float(t_tot) * self.batch_size / N)

        self._run_epochs(train_data, val_data, N)

        return self.history

    def _run_epochs(self, train_data, val_data, N):
        n_batches = N // self.batch_size

        if self.verbose:
            for_epoch = trange(self.n_epochs, leave=True, unit='epoch')

            for epoch in for_epoch:
                self.shuffleData(train_data)

                # Evaluate for saving in history
                train_cost, _, train_acc, val_cost, _, val_acc = self._update_history(train_data, val_data)

                for_epoch.set_description(f'train_cost: {train_cost:.4f}\ttrain_acc: {100*train_acc:.2f}%' + ' | ' +
                                          f'val_cost: {val_cost:.4f}\tval_acc: {100*val_acc:.2f}% ')

                self._run_batches(train_data, n_batches, epoch)
        else:

            for epoch in range(self.n_epochs):
                self.shuffleData(train_data)

                # Evaluate for saving in history
                train_cost, _, train_acc, val_cost, _, val_acc = self._update_history(train_data, val_data)

                self._run_batches(train_data, n_batches, epoch)

    def _run_batches(self, train_data, n_batches, epoch):

        for b in range(n_batches):
            X_batch, Y_batch, y_batch = self._get_mini_batch(train_data, b)

            self._update_eta(b, n_batches, epoch)
            self.history['eta'].append(self.eta)
            self._update_weights(X_batch, Y_batch, y_batch)

    def _update_weights(self, X_batch, Y_batch, y_batch):
        H, P = self.forward_pass(X_batch)
        dL_dW1, dL_db1, dL_dW2, dL_db2 = self.backward_pass(X_batch, Y_batch, H, P)

        if self.check_gradient:
            self._check_gradient_numerically(X_batch, Y_batch, y_batch, N=2, M=20)

        self.W1 -= self.eta * (dL_dW1 + 2 * self.lambda_L2 * self.W1)
        self.b1 -= self.eta * dL_db1
        self.W2 -= self.eta * (dL_dW2 + 2 * self.lambda_L2 * self.W2)
        self.b2 -= self.eta * dL_db2

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

    def _update_history(self, train_data, val_data):
        train_cost, train_loss, train_acc = self.evaluate(train_data[0], train_data[2])
        val_cost, val_loss, val_acc = self.evaluate(val_data[0], val_data[2])

        self.history['train_cost'].append(train_cost)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_cost'].append(val_cost)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

        return train_cost, train_loss, train_acc, val_cost, val_loss, val_acc

    def _update_eta(self, b, batches, epoch):

        t = batches * epoch + b
        index = t % (self.n_s * 2)
        self.eta = self.scheduled_eta[index]

    def split_data(self, train_data, val_split):
        self.shuffleData(train_data)
        train_data_new = []
        val_data = []

        N = train_data[0].shape[1]
        end_train = N - int(N * val_split)

        train_data_new.append(train_data[0][:, :end_train])
        train_data_new.append(train_data[1][:, :end_train])
        train_data_new.append(train_data[2][:end_train])

        val_data.append(train_data[0][:, end_train:])
        val_data.append(train_data[1][:, end_train:])
        val_data.append(train_data[2][end_train:])

        return train_data_new, val_data

    """
    Functions for numerical check of the gradient
    """

    def forward_pass_num(self, X, W1, W2):
        S1 = W1.dot(X) + self.b1
        H = np.maximum(0, S1)
        S = W2.dot(H) + self.b2
        P = softmax(S)
        return H, P

    def compute_cost_num(self, X, y, W1, W2, P=None):
        N = X.shape[1]
        if P is None:
            P = self.forward_pass_num(X, W1, W2)[1]

        loss = -np.log(P[y, np.arange(N)]).mean()
        reg_term = self.lambda_L2 * (np.square(W1).sum() + np.square(W2).sum())

        return loss + reg_term, loss

    def backward_pass_num(self, X, Y, W1, W2, H, P):
        ones_nb = np.ones(X.shape[1])
        c = 1 / X.shape[1]
        G = P - Y
        dL_dW2 = c * G.dot(H.T)
        dL_db2 = c * G.dot(ones_nb).reshape(self.K, 1)
        G = W2.T.dot(G)
        G = G * (H > 0)
        dL_dW1 = c * G.dot(X.T)
        dL_db1 = c * G.dot(ones_nb).reshape(W1.shape[0], 1)

        return dL_dW1, dL_db1, dL_dW2, dL_db2

    def _compute_grad_num_slow(self, X, y, W1, W2, h=1e-5):

        dL_dW1 = np.zeros(W1.shape)
        dL_db1 = np.zeros(self.b1.shape)
        dL_dW2 = np.zeros(W2.shape)
        dL_db2 = np.zeros(self.b2.shape)

        for i in range(len(self.b1)):
            self.b1[i] -= h
            c1 = self.compute_cost_num(X, y, W1, W2)[0]
            self.b1[i] += 2 * h
            c2 = self.compute_cost_num(X, y, W1, W2)[0]
            self.b1[i] -= h
            dL_db1[i] = (c2 - c1) / (2 * h)

        for i in range(len(self.b2)):
            self.b2[i] -= h
            c1 = self.compute_cost_num(X, y, W1, W2)[0]
            self.b2[i] += 2 * h
            c2 = self.compute_cost_num(X, y, W1, W2)[0]
            self.b2[i] -= h
            dL_db2[i] = (c2 - c1) / (2 * h)

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1[i, j] -= h
                c1 = self.compute_cost_num(X, y, W1, W2)[0]
                W1[i, j] += 2 * h
                c2 = self.compute_cost_num(X, y, W1, W2)[0]
                W1[i, j] -= h
                dL_dW1[i, j] = (c2 - c1) / (2 * h)

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2[i, j] -= h
                c1 = self.compute_cost_num(X, y, W1, W2)[0]
                W2[i, j] += 2 * h
                c2 = self.compute_cost_num(X, y, W1, W2)[0]
                W2[i, j] -= h
                dL_dW2[i, j] = (c2 - c1) / (2 * h)

        return dL_dW1, dL_db1, dL_dW2, dL_db2

    def _check_gradient_numerically(self, X_batch, Y_batch, y_batch, N, M):
        """
        :param N: Number of samples to test
        :param M: Number of featuers to test
        """

        H, P = self.forward_pass_num(X_batch[:M, :N], W1=self.W1[:, :M], W2=self.W2)
        dL_dW1_check, dL_db1_check, dL_dW2_check, dL_db2_check = self.backward_pass_num(X_batch[:M, :N],
                                                                                        Y_batch[:, :N],
                                                                                        self.W1[:, :M], self.W2, H,
                                                                                        P)
        dL_dW1_n, dL_db1_n, dL_dW2_n, dL_db2_n = self._compute_grad_num_slow(X_batch[:M, :N], y_batch[:N],
                                                                             self.W1[:, :M], self.W2)

        ok_W1 = self._compute_numerical_error(dL_dW1_n, dL_dW1_check)
        ok_b1 = self._compute_numerical_error(dL_db1_n, dL_db1_check)
        ok_W2 = self._compute_numerical_error(dL_dW2_n, dL_dW2_check)
        ok_b2 = self._compute_numerical_error(dL_db2_n, dL_db2_check)
        print(f'Sanity check: {ok_W1}, {ok_b1}, {ok_W2}, {ok_b2}')

    def _compute_numerical_error(self, A_num, A_check):
        eps = 1e-8
        tolerance_error = 1e-5
        num = np.abs(A_check - A_num)
        den = np.maximum(eps, np.abs(A_num) + np.abs(A_check))
        err = num / den
        max_err = err.max()
        n_ok = (err < tolerance_error).sum()
        p_ok = n_ok / A_num.size * 100

        print(f'Max error: {max_err}\nPercentage of values under max tolerated value: {p_ok}\n' +
              f'eps: {eps}\tMax tolerated error: {tolerance_error}')

        return p_ok
