import numpy as np

from natorch.nn.modules.container import Sequential
from natorch.nn.modules import Module
from natorch.nn import Conv2d, AvgPool2d, Flatten, Dense
from natorch.nn.modules.activation import Tanh, Softmax

class LeNet5(Module): 

    def __init__(self, num_classes : int  = 10):
        
        self.net = Sequential(Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                              Tanh(),
                              AvgPool2d(2, stride=2),
                              Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                              Tanh(),
                              AvgPool2d(2, stride=2),
                              Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
                              Tanh(),
                              Flatten(), 
                              Dense(120, 84), 
                              Tanh(),
                              Dense(84, 10), 
                              Softmax()
                              )

    def forward(self, x):
        return self.net.forward(x)

    def backward(self, grad_out):
        return self.net.backward(grad_out=grad_out)
