import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
import layer
import math
from functools import reduce
from abc import ABC, abstractmethod
from scipy.special import expit


class NeuralNetwork(ABC):
    @abstractmethod
    def compute(self, x):
        pass

    def set_data(self, treining_set, test_set):
        self.treining_set = treining_set
        self.test_set = test_set

    @abstractmethod
    def set_structure(self, structure, attributes_in_data):
        pass


class MultiLayerNetwork(NeuralNetwork):
    def __init__(self, structure, attributes_in_data):
        self.set_structure(structure, attributes_in_data)

    def set_structure(self, structure, attributes_in_data):
        self.layers = []

        def add_layer(input, output, self=self):
            self.layers.append(layer.Layer(input, output, expit))
            return output

        reduce(add_layer, structure, attributes_in_data)

    def compute(self, x):
        return reduce(lambda returned, layer: layer.compute(returned), self.layers, x)

class Kohonen2DNetwork(NeuralNetwork):
    def __init__(self, structure, attributes_in_data):
        self.set_structure(structure, attributes_in_data)

    def set_structure(self, structure, attributes_in_data):
        self.layers = layer.KohonenLayer( attributes_in_data, structure)

#
#     def compute(self, x):
#         return reduce(lambda returned, layer: layer.compute(returned), self.layers, x)
