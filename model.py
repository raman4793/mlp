import numpy as np
from layer import Layer


class Model:
    def __init__(self):
        self.layers = np.array([], dtype=Layer)
        self.inputs = None
        self.targets = None
        self.outputs = None
        self.learning_rate = None
        self.epochs = None
        self.loss_function = None

    def add(self, layer):
        layer.__set_id__(self.layers.size + 1)
        self.layers = np.append(self.layers, layer)

    def train(self, inputs, targets, learning_rate, epochs, loss_function):
        self.__setup__(inputs, targets, learning_rate, epochs, loss_function)
        for epoch in range(epochs):
            guess = self.predict(inputs, all_outs=True)
            guess = guess[-1]
            self.loss_function(self)

    def predict(self, inputs, all_outs=False):
        for layer in self.layers:
            self.outputs = [inputs]
            inputs = layer.process(inputs)
        return self.outputs if all_outs else inputs

    def gradient_descend(self):
        pass

    def __setup__(self, inputs, targets, learning_rate, epochs, loss_function):
        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_function = loss_function


def sigmoid(x, derivative=False):
    if derivative:
        sig = sigmoid(x)
        return 1 * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
