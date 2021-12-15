
import numpy as np


def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid_activation(x)*(1-sigmoid_activation(x))