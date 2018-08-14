import numpy as np
from layer import Layer
from model import Model
import model

inputs = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

outputs = np.array([
    [0],
    [1],
    [1],
    [0]
])

learning_rate = 0.2

epochs = 1

mlp = Model()

activation_function = model.sigmoid
loss_function = Model.gradient_descend

input_layer = Layer(2, activation_function)
mlp.add(input_layer)

hidden_layer = Layer(1, activation_function)
mlp.add(hidden_layer)

mlp.train(inputs, outputs, learning_rate, epochs, loss_function)

print(mlp.predict(inputs))
