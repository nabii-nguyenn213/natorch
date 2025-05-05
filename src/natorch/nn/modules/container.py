import numpy as np
from typing import List
from natorch.nn.modules import Module
# layers : 
from natorch.nn.modules import Dense, AvgPool2d, Conv2d, MaxPool2d, BatchNorm2d
# activation 
from natorch.nn.modules.activation import ReLU, LeakyReLU, Sigmoid, Softmax, Tanh
# loss : 
from natorch.nn.modules.losses import BCELoss, MSELoss

class Sequential(Module):

    def __init__(self, *layers):
        super().__init__()
        self._layers = list(layers) if layers else []

    def add(self, layer):
        self._layers.append(layer)

    def __len__(self):
        return len(self._layers)

    def _parameter(self) -> List: 
        params = []
        for i in self._layers: 
            if i._parameters != {}:
                params.append(i._parameters)
        return params

    def forward(self, x):
        if self._layers == []:
            return 

        output = x
        for layer in self._layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_out):
        grad = grad_out
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    def __repr__(self):
        pass
