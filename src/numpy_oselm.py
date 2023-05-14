
from math import ceil

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(a, b):
    return ((a - b) ** 2).mean(axis=None)


class OS_ELM(object):
    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes):
        # TEMP Implement more activation functions
        self.activation = sigmoid

        # TEMP Implement more loss functions
        self.loss = mean_squared_error

        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes

        self.__is_finished_init_train = False

        # alpha = learning rate
        self.__alpha = np.array([
            [np.random.uniform(-1, 1) for _ in range(self.__n_hidden_nodes)]
            for i in range(self.__n_input_nodes)
        ])
        self.__bias = np.array([
            np.random.uniform(-1, 1)
            for _ in range(self.__n_hidden_nodes)
        ])
        # beta is the weight of precision over recall in F-beta score
        self.__beta = np.zeros(
            shape=[self.__n_hidden_nodes, self.__n_output_nodes])
        # model parameter
        self.__p = np.zeros(
            shape=[self.__n_hidden_nodes, self.__n_hidden_nodes])

    def predict(self, x):
        return np.dot(
            self.activation(np.dot(x, self.__alpha) + self.__bias),
            self.__beta
        )

    def evaluate(self, x, t, metrics=None):
        if metrics is None:
            metrics = ['loss']
        met = []
        for m in metrics:
            if m == 'loss':
                met.append(self.loss(self.predict(x), t))
            elif m == 'accuracy':
                tp = fp = 0
                for i in range(len(x)):
                    if ceil(self.predict(x)[i]) == t[i]:
                        tp += 1
                    else:
                        fp += 1
                met.append((tp / (tp + fp)))
            else:
                return ValueError(f"An unknown metric '{m}' was given.")

        return met

    def init_train(self, x, t):
        if self.__is_finished_init_train:
            raise Exception(
                "The initial training phase has already finished. Please call "
                "'seq_train' method for further training.")
        if len(x) < self.__n_hidden_nodes:
            raise ValueError(
                "In the initial training phase, the number of training samples must be "
                "greater than the number of hidden nodes. But this time "
                f"len(x)={len(x)}, while n_hidden_nodes={self.__n_hidden_nodes}"
            )

        self.__build_init_train_graph(x, t)
        self.__is_finished_init_train = True

    def seq_train(self, x, t):
        if not self.__is_finished_init_train:
            raise Exception(
                "You have not gone through the initial training phase yet. Please "
                "first initialize the model's weights by 'init_train' method before "
                "calling 'seq_train' method."
            )
        self.__build_seq_train_graph(x, t)

    def __build_init_train_graph(self, x, t):
        # hidden layer output matrix
        H = self.activation(np.dot(x, self.__alpha) + self.__bias)
        HT = np.transpose(H)
        HTH = np.dot(HT, H)
        self.__p = np.linalg.inv(HTH)
        pHT = np.dot(self.__p, HT)
        self.__beta = np.dot(pHT, t)
        return self.__beta

    def __build_seq_train_graph(self, x, t):
        # hidden layer output matrix
        h = self.activation(np.dot(x, self.__alpha) + self.__bias)
        ht = np.transpose(h)
        batch_size = x.shape[0]
        i = np.eye(batch_size)
        hp = np.dot(h, self.__p)
        hp_ht = np.dot(hp, ht)
        temp = np.linalg.inv(i + hp_ht)
        pht = np.dot(self.__p, ht)
        self.__p -= np.dot(np.dot(pht, temp), hp)
        pht = np.dot(self.__p, ht)
        hbeta = np.dot(h, self.__beta)
        self.__beta += np.dot(pht, t - hbeta)
        return self.__beta
