import numpy as np
import math

input_angle = 0
input_distance = 1000


X = [[input_angle, input_distance]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation:
    def forward(self, inputs):
        self.output = 1/(1 + np.exp(-inputs))


layer1 = Layer_Dense(2, 16)
layer2 = Layer_Dense(16, 2)

activation1 = Activation()
activation2 = Activation()


    
layer1.forward(X)

activation1.forward(layer1.output)

layer2.forward(activation1.output)

activation2.forward(layer2.output)

output_rotz = activation2.output[0][0] - 0.5
output_speed = activation2.output[0][1]
