import numpy as np
from layer import Layer


class Model:
    def __init__(self):
        self.layers = np.array([], dtype=Layer)
        self.inputs = None
        self.targets = None
        self.outputs = []
        self.learning_rate = None
        self.epochs = None
        self.loss_function = None
        self.batch = None

    def add(self, layer: Layer):
        layer.__set_id__(self.layers.size + 1)
        self.layers = np.append(self.layers, layer)

    def train(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float, epochs: int,
              batch: int, loss_function):
        """
        Trains the neural network
        set batch = 0 if you don't want to specify batch size
        """
        self.__setup__(inputs, targets, learning_rate, epochs, batch, loss_function)
        for epoch in range(epochs):
            print("EPOCH ", epoch)
            self.__step__()

    def predict(self, inputs: np.ndarray):
        """
        Predict method takes an numpy array of inputs/input and returns numpy array of outputs/output
        """
        self.outputs = []
        self.outputs.append(inputs)
        for layer in self.layers:
            inputs = layer.process(inputs)
            self.outputs.append(inputs)
        return inputs

    def __step__(self):
        """
        Predict method takes an numpy array of inputs/input and returns numpy array of outputs/output
        """
        if self.batch == 0 or self.batch == len(self.inputs):
            self.predict(self.inputs)
            self.loss_function(self)
        else:
            for i in range(0, len(self.inputs), self.batch):
                batch_input = self.inputs[i:i+self.batch]
                self.predict(batch_input)
                batch_target = self.targets[i:i+self.batch]
                self.loss_function(self, targets=batch_target)

    def gradient_descend(self, targets=None):
        if targets is None:
            targets = self.targets
        i = self.layers.size
        guess = self.outputs[-1]
        error = targets - guess
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

    def __setup__(self, inputs, targets, learning_rate, epochs, batch, loss_function):
        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch = batch
        self.loss_function = loss_function


def sigmoid(x, derivative=False):
    if derivative:
        x = sigmoid(x)
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
