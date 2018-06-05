import cupy as cp  # biblioteka działająca jak numpy, tylko że wykonująca operacje na GPU na rdzenach CUDA
import numpy as np  # najpopularniejsza pythonowa biblioteka do operacji matematycznych głównie macierzowych
from math import floor, sqrt
from functools import reduce
from abc import ABC, abstractmethod
import helpers


class Wage_initializer(ABC):
    @abstractmethod
    def wage_init(self, number_of_input, number_of_output):
        # number of output == number of neurons
        pass


class Random_wage_init(Wage_initializer):
    def wage_init(self, number_of_input, number_of_output):
        return cp.random.rand(number_of_input, number_of_output)


class Close_to_data_wage_init(Wage_initializer):
    def __init__(self, data, delta = 0.02):
        self.data = data
        self.delta = delta

    def wage_init(self, number_of_input, number_of_output):
        W = list()
        self.number_of_input = number_of_input
        self.number_of_output = number_of_output
        random_indexes = helpers.random_indexes(self.data)[0:number_of_output]
        for random_index in random_indexes:
            W.append(
                list(map(lambda x: float(x), list(self.data[random_index] + self.random_vect())))
            )
        return cp.array(W).transpose()

    def random_vect(self):
        return self.delta * cp.random.rand(self.number_of_input)


# klasa przydatna na wypadek chęci tworzenia sieci MLP
class Layer:
    def __init__(self, number_of_input, number_of_output, activation_fun):
        self.W = cp.random.rand(number_of_input, number_of_output)
        self.activation_fun = activation_fun

    def compute(self, x):
        return self.activation_fun(cp.dot(x, self.W))

    def set_structure(self, number_of_input, number_of_output):
        self.W = cp.random.rand(number_of_input, number_of_output)


class Kohonen2DLayer:
    def __init__(self, number_of_input, number_of_output, wage_initializer=Random_wage_init()):
        # self.distance_dependency_fun = distance_dependency_fun
        self.set_structure(number_of_input, number_of_output, wage_initializer)

    def set_structure(self, number_of_input, number_of_output, wage_initializer=Random_wage_init()):
        # print (wage_initializer)
        self.W = wage_initializer.wage_init(number_of_input, number_of_output)
        print(self.W.shape)
        positions = list([])
        floor_sqrt_output = int(floor(sqrt(number_of_output)))
        for number_x in range(0, floor_sqrt_output):
            for number_y in range(0, floor_sqrt_output):
                positions.append([number_x, number_y])
        for number_y in range(0, number_of_output - floor_sqrt_output ** 2):
            positions.append([floor_sqrt_output, number_y])
        self.positions = cp.array(positions)

    # licz produkt macierzowy, właściwie zbędna funkcja
    def compute(self, x):
        return cp.dot(x, self.W)

    # wyszukuje BMU - neuronu najsilniej odpowiadającego. Korzysta z określania odległości jako kąty
    def find_best_neuron_by_dot_product(self, x):
        neurons_output = cp.dot(x, self.W).transpose()
        output_max = neurons_output[0]
        it_max = 0
        for it, product in enumerate(neurons_output):
            if abs(product) > abs(output_max):
                it_max = it
                output_max = product
        return it_max

    # wyszukuje BMU - neuronu najsilniej odpowiadającego
    def find_best_neuron_by_cartesian_distance(self, x):
        W = self.W.transpose()
        # funkcja ta nie jest dokładnie dystansem kartezjanskim ale zachowuje jego monotoniczność
        return reduce(lambda max, pretender: pretender if pretender[0] > max[0] else max,
                      map(lambda w, it, x=x: (abs(cp.sum(w - x)), it), W, range(len(W))))[1]

    # zakłada prostokątną architekture sieci, i liczy odległość między neuronami w "przestrzeni neuronów"
    def cartesian_distance_between_neurons(self, it_1, it_2):  # dla sieci o topoligi prostokątnej
        dist_x = (self.positions[it_1][0] - self.positions[it_2][0])
        dist_y = (self.positions[it_1][1] - self.positions[it_2][1])
        return sqrt(dist_x ** 2 + dist_y ** 2)

    def distance_between_neurons_as_vectors(self, it_1, it_2):
        W_1 = self.W[it_1]
        W_2 = self.W[it_2]
        # return sqrt(reduce(lambda sum, w: sum + w ** 2, W_1 - W_2))
        return sqrt(cp.sum((W_1 - W_2) ** 2))

    def number_of_neurons(self):
        return len(self.W.transpose())
