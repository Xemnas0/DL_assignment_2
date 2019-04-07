from matplotlib import gridspec
import numpy as np
from utilities import readDataset, normalize
from Model import Model
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def plot_history(history, GDparams):
    fig = plt.figure(1, figsize=(12, 5))
    gs1 = gridspec.GridSpec(1, 2)
    ax = [fig.add_subplot(subplot) for subplot in gs1]

    ax[0].set_title('Cost')
    ax[0].plot(history['train_loss'], sns.xkcd_rgb["pale red"], label='Train')
    ax[0].plot(history['val_loss'], sns.xkcd_rgb["denim blue"], label='Val')
    ax[0].axhline(history['test_loss'], color=sns.xkcd_rgb["medium green"], label='Test')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    ax[1].set_title('Accuracy')
    ax[1].plot(history['train_acc'], sns.xkcd_rgb["pale red"], label='Train')
    ax[1].plot(history['val_acc'], sns.xkcd_rgb["denim blue"], label='Val')
    ax[1].axhline(history['test_acc'], color=sns.xkcd_rgb["medium green"], label='Test')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()

    main_title = 'batch_size={0}, $\eta_{{min}}={1}$, $\eta_{{max}}={2}$, $n_s={3}$'.format(
        GDparams['batch_size'], GDparams['eta_min'], GDparams['eta_max'], GDparams['n_s']
    ) + ', $\lambda={0}$, n_epochs={1}'.format(GDparams['lambda_L2'], GDparams['n_epochs'])
    fig.suptitle(main_title, fontsize=18)

    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    fig.show()


if __name__ == '__main__':

    # Loading datasets
    load_full_data = True
    if load_full_data:
        X_train, Y_train, y_train = readDataset('Datasets/cifar-10-batches-py/data_batch_1')
        for i in range(2,6):
            X, Y ,y = readDataset(f'Datasets/cifar-10-batches-py/data_batch_{i}')
            X_train = np.hstack((X_train, X))
            Y_train = np.hstack((Y_train, Y))
            y_train = np.hstack((y_train, y))
        X_test, Y_test, y_test = readDataset('Datasets/cifar-10-batches-py/test_batch')

    else:
        X_train, Y_train, y_train = readDataset('Datasets/cifar-10-batches-py/data_batch_1')
        X_val, Y_val, y_val = readDataset('Datasets/cifar-10-batches-py/data_batch_2')
        X_test, Y_test, y_test = readDataset('Datasets/cifar-10-batches-py/test_batch')

    # Data characteristics
    d, N = X_train.shape
    K = y_train.max() + 1

    # Gradient Descent parameters
    GDparams = {'batch_size': 100,
                'eta_min': 1e-5,
                'eta_max': 1e-1,
                'n_s': 800,
                'n_epochs': 10,
                'lambda_L2': 0.0001}

    # Normalization of the dataset
    X_train, mean, std = normalize(X_train)
    if not load_full_data:
        X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    model = Model()
    model.compile(d, K, m=50)

    if load_full_data:
        history = model.fit((X_train, Y_train, y_train), GDparams, val_split=0.1)
    else:
        history = model.fit((X_train, Y_train, y_train), GDparams, val_data=(X_val, Y_val, y_val))

    history['test_loss'], history['test_acc'] = model.evaluate(X_test, y_test)
    print('Test loss: {0:.4f}\tTest acc: {1:.2f}%'.format(history['test_loss'], history['test_acc'] * 100))
    plot_history(history, GDparams)
