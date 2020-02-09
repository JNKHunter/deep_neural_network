import pytest
from dnn.dnn_utils import *

@pytest.fixture(scope="module")
def input_layer():
    return [1, 5, 9]

@pytest.fixture(scope="module")
def layer_config(input_layer):
    return [len(input_layer), 4, 5]

@pytest.fixture(scope="module")
def layers(layer_config):
    return create_layers(layer_config)

def test_create_layers(layers):

    assert(layers['W1'].shape == (4, 3))
    assert(layers['b1'].shape == (4, 1))
    assert(layers['W2'].shape == (5, 4))
    assert(layers['b2'].shape == (5, 1))

def test_layer_forward(layers):
    Z, cache = layer_forward(input_layer, layers['W1'], layers['b1'])

