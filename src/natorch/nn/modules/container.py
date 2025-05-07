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
        self._reinitialize_params()

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

    def _reinitialize_params(self) -> None:
        print("Re-initialize parameters")
        activation_name = ["ReLU", "LeakyReLU", "Sigmoid", "Softmax", "Tanh",
                           "Conv2d", "ConvTranspose2d"]

        for i in range(len(self._layers)-1):
            current_class = self._layers[i].__class__.__name__
            if hasattr(current_class, 'nonlinearity'):
                name_next_layer = self._layers[i + 1].__class__.__name__ 
                if name_next_layer in activation_name:
                    if name_next_layer == "ReLU":
                        self._layers[i].nonlinearity = 'relu'
                    elif name_next_layer == "LeakyReLU":
                        self._layers[i].nonlinearity = 'leakyrelu'
                    elif name_next_layer == "Sigmoid":
                        self._layers[i].nonlinearity = 'sigmoid'
                    elif name_next_layer == "Softmax":
                        self._layers[i].nonlinearity = 'softmax'
                    elif name_next_layer == "Tanh":
                        self._layers[i].nonlinearity = 'tanh'
                    elif name_next_layer == "Conv2d":
                        self._layers[i].nonlinearity = 'conv2d'
                    elif name_next_layer == "ConvTranspose2d":
                        self._layers[i].nonlinearity = 'conv_transpose2d'
                else:
                    continue
                
                self._layers[i].initialize_params()
                self._layers[i]._update_params()

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
