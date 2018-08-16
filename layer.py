import numpy as np


class Layer:
    # Initialize an emtpy layer with no weights
    def __init__(self, number_of_neurons: int, activation_function, inputs=None):
        """
        Create a layer with Layer(number of neurons, activation function)
        """
        self.uid = None
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function
        self.inputs = inputs
        self.shape = None
        self.data_shape = None
        self.weights = None
        self.__set_weights()

    def process(self, inputs):
        self.inputs = inputs
        self.__set_weights()
        out = inputs.dot(self.weights)
        out = self.activation_function(out)
        return out

    def __set_weights(self):
        if self.weights is None and self.inputs is not None:
            self.data_shape = self.inputs.shape
            self.shape = (self.data_shape[-1], self.number_of_neurons)
            self.weights = np.random.random(self.shape)

    def __set_id__(self, uid):
        self.uid = uid
