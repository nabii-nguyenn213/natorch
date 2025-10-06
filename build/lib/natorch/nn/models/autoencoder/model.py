from natorch.nn.functional import conv2d
import numpy as np

from natorch.nn.modules import Module
from natorch.nn.modules.container import Sequential
from natorch.nn import Conv2d, BatchNorm2d, MaxPool2d, Flatten, Dense, UnFlatten, ConvTranspose2d
from natorch.nn.modules.activation import LeakyReLU, Sigmoid
from natorch.nn.init import kaiming_normal_, kaiming_uniform_, xavier_normal_, xavier_uniform_

class AutoEncoder(Module):
    
    def __init__(self, latent_dim : int = 10, negative_slope : float = 0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.negative_slope = negative_slope

        self.net = Sequential(
                              # ENCODER 
                              Conv2d(1, 16, kernel_size=3, stride=1, padding=1, initialization='kaiming_normal'), 
                              LeakyReLU(negative_slope=self.negative_slope),
                              BatchNorm2d(16), 
                              MaxPool2d(2, stride=2), 
                              Conv2d(16, 32, kernel_size=3, stride=1, padding=1, initialization='kaiming_normal'), 
                              LeakyReLU(negative_slope=self.negative_slope), 
                              BatchNorm2d(32), 
                              MaxPool2d(2, stride=2), 
                              Flatten(), 
                              Dense(1568, self.latent_dim, initialization='kaiming_normal'),
                              LeakyReLU(negative_slope=self.negative_slope), 

                              # DECODER
                              Dense(self.latent_dim, 1568, initialization='kaiming_normal'), 
                              LeakyReLU(negative_slope=self.negative_slope), 
                              UnFlatten(32, 7, 7),
                              ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, initialization='kaiming_normal', output_padding=1), 
                              LeakyReLU(negative_slope=self.negative_slope), 
                              BatchNorm2d(16), 
                              ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, initialization='kaiming_normal', output_padding=1), 
                              LeakyReLU(negative_slope=self.negative_slope), 
                              BatchNorm2d(8), 
                              Conv2d(8, 1, kernel_size=3, stride=1, padding=1, initialization='xavier_normal'), 
                              Sigmoid()
                              )

    def forward(self, x):
        return self.net.forward(x)

    def backward(self, grad_out):
        return self.net.backward(grad_out=grad_out)
