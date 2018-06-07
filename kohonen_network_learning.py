import numpy as np
from scipy.stats import norm  # potrzebne do prostej i szybkiej implementacji funkcji Gaussa
import copy  # używane do głębokich kopii sieci
import helpers
from abc import ABC, abstractmethod


# funktor pozwalający na ustawienie i potem zawsze zwracanie wartości funkcji Gaussa dla nastaw i danego punktu,
# zawsze zwraca wartość z 1 w mean - służy do liczenia wpływu odległości
class Gauss:
    def __init__(self, mean, std_dev):
        self.rv = norm(loc=mean, scale=std_dev)
        self.max_rv = self.rv.pdf(mean)

    def val(self, x):
        return self.rv.pdf(x) / self.max_rv


# Funktory używane do uczenia zmiany parametrów uczenia sieci
class DecreasingFunction(ABC):
    @abstractmethod
    def val(self, x):
        pass


class AlphaFun(DecreasingFunction):
    def __init__(self, alpha_0=0.01, t=0.7):
        self.t = t
        self.a_0 = alpha_0

    def val(self, x):
        return self.a_0 * np.exp(self.t - self.t / (1 - x)) if x != 1 else 0.


class DecreasingDistFun(AlphaFun):
    def __init__(self, d_0=20., t=1.2):
        super().__init__(alpha_0=d_0, t=t)


# sieć SOM z funkcją Gaussa jako funkcją odległości i ustaloną funkcją dla learning rate, w późniejszych etapach projektu
# ulegnie refaktoryzacji dla większej elastyczności (możliwość wyboru poszczególnych funkcji i ich parametrów)
class KohonenNetworkGauss_nbh:
    def __init__(self, data, network,
                 alpha_fun=AlphaFun(),
                 dist_fun=DecreasingDistFun(),
                 if_by_dot_product=False):
        self.data = data
        self.network = network
        self.by_dot_product = if_by_dot_product
        self.alpha_fun = alpha_fun
        self.dist_fun = dist_fun

    # główna część - algorytm uczenia się
    # BMU - indeks najsilniej odpowiadającego neuronu
    def learning(self, number_of_epochs):

        # służy do zapisywania stanów sieci w kolejnych epokach do późniejszej obserwacji
        remember_network = [copy.deepcopy(self.network), ]

        for epoch in range(number_of_epochs):
            for index in helpers.random_indexes(self.data):
                # print(self.data[index])
                x = self.data[index].reshape(1, len(self.data[0]))
                s = epoch / number_of_epochs
                if (self.by_dot_product):
                    BMU = self.network.layers.find_best_neuron_by_dot_product(x)
                else:
                    BMU = self.network.layers.find_best_neuron_by_cartesian_distance(x)
                self.nbh_fun(x, BMU, s)
                # print(BMU)
                # print(self.network.layers.W.transpose()[0])
            print("epoch: " + str(epoch))
            print(self.network.layers.W.transpose()[0])
            remember_network.append(copy.deepcopy(self.network))
        return remember_network

    # nazwa jest niedokładna. Jest to funkcja sąsiedstwa, oraz algorytm uczenia neuronu na jej podstawie:
    # x - wektor wejściowy
    # BMU - indeks neuronu najmocniej odpowiadającego
    # s - zmienna informująca o tym ile zostało do zakończenia epok
    # dist_0 - zasięg w "neuronach" początkowy - zasięg definiowany jako odchylenie standardowe funkcji Gaussa
    def nbh_fun(self, x, BMU, s):
        layer = self.network.layers
        W = layer.W.transpose()
        dist_0_act = self.dist_fun.val(s)
        # print(dist_0)
        gauss = Gauss(0., dist_0_act)  # uzywamy zwyklego Gaussa1D do policzenia "wzmocnienia" w funkcji odleglosci
        for v in range(layer.number_of_neurons()):
            dist = layer.cartesian_distance_between_neurons(BMU, v)
            k = gauss.val(dist)
            W[v] = W[v] + (self.alpha_fun.val(s) * k * (x[0] - W[v]))
