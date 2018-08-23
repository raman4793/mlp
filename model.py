import numpy as np
from layer import Layer
import pickle


class Model:
    def __init__(self):
        self.layers = np.array([], dtype=Layer)
        self.inputs = None
        self.targets = None
        self.outputs = []
        self.learning_rate = None
        self.epochs = None
        self.loss_function = None

    def add(self, layer: Layer):
        layer.__set_id__(self.layers.size + 1)
        self.layers = np.append(self.layers, layer)

    def train(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float, epochs: int,
              loss_function):
        self.__setup__(inputs, targets, learning_rate, epochs, loss_function)
        for epoch in range(epochs):
            print("EPOCH ", epoch)
            self.predict(inputs)
            self.loss_function(self)

    def predict(self, inputs: np.ndarray):
        """
        Predict method takes an numpy array of inputs/input and returns numpy array of outputs/output
        """
        if self.inputs is None:
            print("The neural network is not yet defined")
        else:
            self.outputs = []
            self.outputs.append(inputs)
            for layer in self.layers:
                inputs = layer.process(inputs)
                self.outputs.append(inputs)
            return inputs

    def gradient_descend(self):
        i = self.layers.size
        guess = self.outputs[-1]
        error = self.targets - guess
        error = error * self.learning_rate
        print("Total Error = ", np.mean(np.abs(error)))
        for layer, guess in zip(reversed(self.layers), reversed(self.outputs)):
            delta = error * layer.activation_function(guess, derivative=True)
            error = delta.dot(layer.weights.T)
            if i > 0:
                previous_output = self.outputs[i - 1]
                weight_adjust = previous_output.T.dot(delta)
                layer.weights += weight_adjust
            i -= 1

    def save(self, file_name):
        model_file = open(file_name, 'wb')
        pickle.dump(self, model_file)

    @staticmethod
    def load(file_name):
        try:
            model_file = open(file_name, 'rb')
            return pickle.load(model_file)
        except FileNotFoundError:
            print("The model file {} was not found".format(file_name))
            return Model()

    def __setup__(self, inputs, targets, learning_rate, epochs, loss_function):
        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_function = loss_function


def sigmoid(x, derivative=False):
    if derivative:
        x = sigmoid(x)
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
