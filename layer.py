import cupy as cp
import numpy as np
from math import floor, sqrt
from functools import reduce


class Layer:
    def __init__(self, number_of_input, number_of_output, activation_fun):
        self.W = cp.random.rand(number_of_input, number_of_output)
        self.activation_fun = activation_fun

    def compute(self, x):
        return self.activation_fun(cp.dot(x, self.W))

    def set_structure(self, number_of_input, number_of_output):
        self.W = cp.random.rand(number_of_input, number_of_output)


class Kohonen2DLayer:
    def __init__(self, number_of_input, number_of_output, distance_dependency_fun):
        self.distance_dependency_fun = distance_dependency_fun
        self.set_structure(number_of_input, number_of_output)

    def set_structure(self, number_of_input, number_of_output):
        self.W = cp.random.rand(number_of_input, number_of_output)
        self.positions = cp.array([[]])
        floor_sqrt_output = floor(sqrt(number_of_output))
        for number_x in cp.arange(0, floor_sqrt_output):
            for number_y in cp.arange(0, floor_sqrt_output):
                self.positions = self.positions.concatenate(cp.array([number_x, number_y]))
        for number_y in cp.arange(0, number_of_output - floor_sqrt_output ** 2):
            self.positions = self.positions.concatenate(cp.array([floor_sqrt_output + 1, number_y]))

    def compute(self, x):
        return cp.dot(x, self.W)

    def find_best_neuron_by_dot_product(self, x):
        neurons_output = cp.dot(x, self.W)
        output_max = neurons_output[0]
        it_max = 0
        for it, product in enumerate(neurons_output):
            if abs(product) > abs(output_max):
                it_max = it
                output_max = product
        return it_max

    def find_best_neuron_by_cartesian_distance(self, x):  # potencjalnie wolne rozwiązanie w pythonie
        return reduce(lambda max, pretender: pretender if pretender[0] > max[0] else max,
                      map(lambda w, it: (abs(sum(w - x)), it), self.W, range(len(self.W))))[1]

    def distance_between_neurons(self, it_1, it_2):  # dla sievi o topoligi prostokątnej
        dist_x = (self.positions[it_1][0] - self.positions[it_2][0])
        dist_y = (self.positions[it_1][1] - self.positions[it_2][1])
        return sqrt(dist_x ** 2 + dist_y ** 2)

    def distance_between_neurons_as_vectors(self, it_1, it_2):
        W_1 = self.W[it_1]
        W_2 = self.W[it_2]
        return sqrt(reduce(lambda sum, w: sum + w ** 2, W_1 - W_2))
