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

def test_create_layers(layers):

    assert(layers['W1'].shape == (4, 3))
    assert(layers['b1'].shape == (4, 1))
    assert(layers['W2'].shape == (5, 4))
    assert(layers['b2'].shape == (5, 1))

def test_layer_forward(layers, input_layer):
    Z, cache = layer_forward(input_layer, layers['W1'], layers['b1'])
    print("A-------------")
    print(input_layer)
    print("W1-------------")
    print(layers['W1'])
    print("b1-------------")
    print(layers['b1'])
    print("Z--------------")
    print(Z)
    assert(Z.shape == (4, 4))