import layer  # plik widoczny niżej
from functools import reduce
from abc import ABC, abstractmethod  # funkcje abstrakcyjne w Pythonie


class NeuralNetwork(ABC):

    def set_data(self, treining_set, test_set):
        self.treining_set = treining_set
        self.test_set = test_set

    @abstractmethod
    def set_structure(self, structure, attributes_in_data):
        pass


# sieć typu MLP bez algorytmu uczenia
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

#Sieć typu SOM, ma zaimplementowany algorytm uczenia jej używający
class Kohonen2DNetwork(NeuralNetwork):
    def __init__(self, structure, attributes_in_data):
        self.set_structure(structure, attributes_in_data)

    def set_structure(self, structure, attributes_in_data):
        self.layers = layer.Kohonen2DLayer(attributes_in_data, structure)

#
#     def compute(self, x):
#         return reduce(lambda returned, layer: layer.compute(returned), self.layers, x)
