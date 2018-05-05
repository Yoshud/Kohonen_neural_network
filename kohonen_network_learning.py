import numpy as np
from scipy.stats import norm
# import cupy as cp
# import matplotlib.pyplot as plt
# import pandas as pd
# import layer
# import math
# import neural_network
# from functools import reduce
# from abc import ABC, abstractmethod
# from scipy.special import expit


class Gauss:
    def __init__(self, mean, std_dev):
        self.rv = norm(loc=mean, scale=std_dev)
        self.max_rv = self.rv.pdf(mean)

    def val(self, x):
        return self.rv.pdf(x) / self.max_rv


class KohonenNetworkLearning:
    def __init__(self):
        pass

    def learning(self):
        pass

    def nbh_gaussian_fun(self, BMU, v, s):
        pass

    def nbh_simple_fun(self, BMU, v, s):
        pass

    def alpha_linear_fun(self, s):
        pass

    def alpha_exponential_fun(self, s):
        pass


class KohonenNetworkGauss_nbh:
    def __init__(self, data, network):
        self.data = data
        self.network = network

    def randomize_data_indexes(self):
        one = list(range(0, len(self.data)))
        second = list(np.random.rand(len(self.data)))
        x = np.concatenate((np.array([one]), np.array([second])), axis=0)
        lista = []
        for el in x.transpose():
            lista.append((el[0], el[1]))
        lista.sort(key=lambda x: x[1])

        return list(map(lambda el: int(el[0]), lista))

    def learning(self, number_of_epochs):
        for epoch in range(number_of_epochs):
            for index in self.randomize_data_indexes():
                x = self.data[index].reshape(1, len(self.data[0]))
                s = epoch / number_of_epochs
                BMU = self.network.layers.find_best_neuron_by_cartesian_distance(x)
                self.nbh_fun(x, BMU, s, 10)
            print("epoch: " + str(epoch))
            print(self.network.layers.W.transpose()[0])
    def alpha_fun(self, s, a0=1., t=2.):
        if (s != 1):
            return a0 * np.exp(t - t / (1 - s))
        else:
            return 0.

    def deacresing_dist_fun(self, dist0, s, t=2.):
        return self.alpha_fun(s, dist0, t)

    def nbh_fun(self, x, BMU, s, dist_0):
        layer = self.network.layers
        W = layer.W.transpose()
        dist_0_act = self.deacresing_dist_fun(dist_0, s)
        gauss = Gauss(0., dist_0_act)  # uzywamy zwyklego Gaussa1D do policzenia "wzmocnienia" w funkcji odleglosci
        for v in range(layer.number_of_neurons()):
            dist = layer.cartesian_distance_between_neurons(BMU, v)
            k = gauss.val(dist)
            W[v] = W[v] + self.alpha_fun(s, 0.2, 2.) * k * (x - W[v])[0]
