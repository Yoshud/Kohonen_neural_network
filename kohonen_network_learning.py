import numpy as np
from scipy.stats import norm
import copy


# funktor pozwalający na ustawienie i potem zawsze zwracanie wartości funkcji Gaussa dla nastaw i danego punktu,
# zawsze zwraca wartość z 1 w mean - służy do liczenia wpływu odległości
class Gauss:
    def __init__(self, mean, std_dev):
        self.rv = norm(loc=mean, scale=std_dev)
        self.max_rv = self.rv.pdf(mean)

    def val(self, x):
        return self.rv.pdf(x) / self.max_rv


# sieć SOM z funkcją Gaussa jako funkcją odległości i ustaloną funkcją dla learning rate, w późniejszych etapach projektu
# ulegnie refaktoryzacji dla większej elastyczności (możliwość wyboru poszczególnych funkcji i ich parametrów)
class KohonenNetworkGauss_nbh:
    def __init__(self, data, network, if_by_dot_product=True):
        self.data = data
        self.network = network
        self.by_dot_product = if_by_dot_product

    # funcja zwracająca indeksy danych w losowej kolejności by móc później "karmić" sieć danymi w losowej kolejności
    def randomize_data_indexes(self):
        one = list(range(0, len(self.data)))
        second = list(np.random.rand(len(self.data)))
        x = np.concatenate((np.array([one]), np.array([second])), axis=0)
        lista = []
        for el in x.transpose():
            lista.append((el[0], el[1]))
        lista.sort(key=lambda x: x[1])

        return list(map(lambda el: int(el[0]), lista))

    # główna część - algorytm uczenia się
    # BMU - indeks najsilniej odpowiadającego neuronu
    def learning(self, number_of_epochs):

        # służy do zapisywania stanów sieci w kolejnych epokach do późniejszej obserwacji
        remember_network = [copy.deepcopy(self.network), ]

        for epoch in range(number_of_epochs):
            for index in self.randomize_data_indexes():
                x = self.data[index].reshape(1, len(self.data[0]))
                s = epoch / number_of_epochs
                if (self.by_dot_product):
                    BMU = self.network.layers.find_best_neuron_by_dot_product(x)
                else:
                    BMU = self.network.layers.find_best_neuron_by_cartesian_distance(x)
                # print(BMU)
                self.nbh_fun(x, BMU, s, 30)
            print("epoch: " + str(epoch))
            print(self.network.layers.W.transpose()[0])
            remember_network.append(copy.deepcopy(self.network))
        return remember_network

    # funkcja malejąca do learning rate
    def alpha_fun(self, s, a0=1., t=2.0):
        if (s != 1):
            return a0 * np.exp(t - t / (1 - s))
        else:
            return 0.

    # funkcja ustalająca malejący się rozmiar zasięgu funkcji sąsiedstwa
    def deacresing_dist_fun(self, dist0, s, t=1.2):
        return self.alpha_fun(s, dist0, t)

    # nazwa jest nie dokładna. Jest to funkcja sąsiedstwa, oraz algorytm uczenia neronu na jej podstawie:
    # x - wektor wejściowy
    # BMU - indeks neuronu najmocniej odpowiadającego
    # s - zmienna informująca o tym ile zostało do zakończenia epok
    # dist_0 - zasięg w "neuronach" początkowy - zasięg definiowany jako odchylenie standardowe funkcji Gaussa
    def nbh_fun(self, x, BMU, s, dist_0):
        layer = self.network.layers
        W = layer.W.transpose()
        dist_0_act = self.deacresing_dist_fun(dist_0, s)
        gauss = Gauss(0., dist_0_act)  # uzywamy zwyklego Gaussa1D do policzenia "wzmocnienia" w funkcji odleglosci
        for v in range(layer.number_of_neurons()):
            dist = layer.cartesian_distance_between_neurons(BMU, v)
            k = gauss.val(dist)
            W[v] = W[v] + self.alpha_fun(s, 0.2, 0.7) * k * (x - W[v])[0]
