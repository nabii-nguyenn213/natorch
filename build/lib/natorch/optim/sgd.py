from natorch.nn.parameter import Parameter
import numpy as np
from typing import List
from natorch.nn.modules import Module

class SGD:

    def __init__(self, parameters : List, lr = 1e-3):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters: 
            for k, v in p.items():
                p[k].zero_grad()

    def step(self) -> None:
        for p in self.parameters:
            for k, v in p.items():
                p[k].data -= self.lr * p[k].grad

