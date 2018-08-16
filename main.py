import numpy as np
from layer import Layer
from model import Model
import model

# Inputs to train
inputs = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])
# Labels or expected output for each inputs
outputs = np.array([
    [0],
    [1],
    [1],
    [0]
])
# Learning rate for the model
learning_rate = 0.2
# epochs for training or in simple words no of training iteration
epochs = 50000
# Creating a model object
mlp = Model()
# setting the activation function as sigmoid method defined in model package
activation_function = model.sigmoid
# setting loss function as gradient descend for the model
loss_function = Model.gradient_descend
# creating an layer with 4 neurons and activation function as sigmoid
layer = Layer(4, activation_function)
# adding the layer object to our model
mlp.add(layer)
# creating another layer with 1 neuron for output
layer = Layer(1, activation_function)
# adding the layer to the model
mlp.add(layer)
# at this point we hav a neural network with 2 hidden neurons and one output neuron
# calling train method on the model object with inputs, outputs, the learning rate,
# epochs and the loss function as parameters
mlp.train(inputs, outputs, learning_rate, epochs, loss_function)
# after training we can call predict method on our model object with inputs as parameters
print(mlp.predict(inputs))
