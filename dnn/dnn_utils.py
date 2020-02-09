import numpy as np


def load_data():
    return [1, 3, 5, 7]


def create_layers(layer_dims):
    """
    This function outputs the weights and biases for each layer.

    Args:
      layer_dims (int[]): The number of dimensions in each layer
    Returns:
      Dict: Weights and biases labeled W + L
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def layer_forward(A, W, b):
    """
    This function returns the linear calculation for the layer.

    Args:
      A array: Activations of previous layer
      W W(l) x A matrix representing weights
      b W(l) array of biases
    Returns:
      Dict: Weights and biases labeled W + L
    """
    cache = (A, W, b)
    Z = np.dot(W, A) + b
    return Z, cache

