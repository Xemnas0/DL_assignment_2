from matplotlib import gridspec
import numpy as np
from utilities import readDataset, normalize
from Model import Model
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm

sns.set()

task = {
    'partial_data': True,
    'lambda_search': False
}

# Gradient Descent parameters
GDparams = {'batch_size': 100,
            'eta_min': 1e-5,
            'eta_max': 1e-1,
            'n_s': 900,
            'n_cycles': 2,
            'lambda_L2': 0.01,
            'verbose': False}
# Lambda search
n_lambda_search = 100

val_split = 0.1


def plot_history(history, GDparams):
    fig = plt.figure(1, figsize=(14, 4))
    gs1 = gridspec.GridSpec(1, 3)
    ax = [fig.add_subplot(subplot) for subplot in gs1]

    for i, (title, metric) in enumerate(zip(['Cost', 'Loss', 'Accuracy'], ['cost', 'loss', 'acc'])):
        ax[i].set_title(title)
        ax[i].plot(history[f'train_{metric}'], sns.xkcd_rgb["pale red"], label='Train')
        ax[i].plot(history[f'val_{metric}'], sns.xkcd_rgb["denim blue"], label='Val')
        ax[i].axhline(history[f'test_{metric}'], color=sns.xkcd_rgb["medium green"], label='Test')
        ax[i].text(0.5, history[f'test_{metric}'],
                   '{0:.4f}'.format(history[f'test_{metric}']), fontsize=12, va='center', ha='center',
                   backgroundcolor='w')
        ax[i].set_ylim(bottom=0)
        ax[i].set_xlabel('Epochs')
        ax[i].legend()

    main_title = 'batch_size={0}, $\eta_{{min}}={1}$, $\eta_{{max}}={2}$, $n_s={3}$'.format(
        GDparams['batch_size'], GDparams['eta_min'], GDparams['eta_max'], GDparams['n_s']
    ) + ', $\lambda={0}$, n_cycles={1}'.format(GDparams['lambda_L2'], GDparams['n_cycles'])
    fig.suptitle(main_title, fontsize=18)

    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    fig.show()

    fig.savefig('second_experiment.png'.format(GDparams['lambda_L2']))

    plt.plot(history['eta'])
    plt.show()


"""
Section for single experiment
"""
if __name__ == '__main__' and task['partial_data']:

    # Loading datasets
    load_full_data = False
    if load_full_data:
        X_train, Y_train, y_train = readDataset('Datasets/cifar-10-batches-py/data_batch_1')
        for i in range(2, 6):
            X, Y, y = readDataset(f'Datasets/cifar-10-batches-py/data_batch_{i}')
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

    # Normalization of the dataset
    X_train, mean, std = normalize(X_train)
    if not load_full_data:
        X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    model = Model()
    model.compile(d, K, m=50)

    if load_full_data:
        history = model.fit((X_train, Y_train, y_train), GDparams, val_split=val_split)
    else:
        history = model.fit((X_train, Y_train, y_train), GDparams, val_data=(X_val, Y_val, y_val))

    history['test_cost'], history['test_loss'], history['test_acc'] = model.evaluate(X_test, y_test)
    print('Test cost: {0:.4f}\tTest acc: {1:.2f}%'.format(history['test_cost'], history['test_acc'] * 100))
    plot_history(history, GDparams)

"""
Section for search of lambda
"""
if __name__ == '__main__' and task['lambda_search']:

    # Loading datasets
    load_full_data = True
    if load_full_data:
        X_train, Y_train, y_train = readDataset('Datasets/cifar-10-batches-py/data_batch_1')
        for i in range(2, 6):
            X, Y, y = readDataset(f'Datasets/cifar-10-batches-py/data_batch_{i}')
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

    # Normalization of the dataset
    X_train, mean, std = normalize(X_train)
    if not load_full_data:
        X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    l_min = -5
    l_max = -1
    stats_lambdas = {'lambda': [], 'val_acc': []}
    for i in tqdm(range(n_lambda_search)):
        l = l_min + (l_max - l_min) * np.random.rand()
        GDparams['lambda_L2'] = 10 ** l
        # print(GDparams['lambda_L2'])
        model = Model()
        model.compile(d, K, m=50)

        if load_full_data:
            history = model.fit((X_train, Y_train, y_train), GDparams, val_split=val_split)
        else:
            history = model.fit((X_train, Y_train, y_train), GDparams, val_data=(X_val, Y_val, y_val))

        stats_lambdas['lambda'].append(GDparams['lambda_L2'])
        stats_lambdas['val_acc'].append(history['val_acc'][-1])

        # history['test_cost'], history['test_loss'], history['test_acc'] = model.evaluate(X_test, y_test)
        # print('Test loss: {0:.4f}\tTest acc: {1:.2f}%'.format(history['test_loss'], history['test_acc'] * 100))
        # plot_history(history, GDparams)

    pickle_out = open("stats_lambdas.pickle", "wb")
    pickle.dump(stats_lambdas, pickle_out)
    pickle_out.close()

    fig, ax = plt.subplots()
    ax.plot(stats_lambdas['lambda'], stats_lambdas['val_acc'])
    fig.show()
    fig.savefig('lambda_search_{0}_{1}.png'.format(l_min, l_max))
