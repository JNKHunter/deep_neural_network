import pytest
from dnn.dnn_utils import *

@pytest.fixture(scope="module")
def input_layer():
    return np.array([[1, 5, 9], [2, 1, 9], [4, 1, 2], [3, 6, 4]]).T


@pytest.fixture(scope="module")
def layer_config(input_layer):
    return [3, 4, 5]

@pytest.fixture(scope="module")
def layers(layer_config):
    return create_layers(layer_config)

@pytest.fixture(scope="module")
def Z_and_cache(layers, input_layer):
    return layer_forward(input_layer, layers['W1'], layers['b1'])

def test_create_layers(layers):

    assert(layers['W1'].shape == (4, 3))
    assert(layers['b1'].shape == (4, 1))
    assert(layers['W2'].shape == (5, 4))
    assert(layers['b2'].shape == (5, 1))

def test_calc_activation(Z_and_cache):
    Z, cache = Z_and_cache
    assert(Z.shape == (4, 4))

    s_calc, cache = calc_activation(Z, sigmoid)
    r_calc, cache = calc_activation(Z, relu)

    assert(s_calc.shape == (4, 4))
    assert(r_calc.shape == (4, 4))

def test_linear_activation_forward(layers, input_layer):
    A, cache = linear_activation_forward(input_layer, layers['W1'], layers['b1'], sigmoid)
    assert(A.shape == (4, 4))

