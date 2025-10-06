import numpy as np
from rich.console import Console
from rich.table import Table
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
        self._caches_input = []
        self._caches_output = []

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
        # print("Re-initialize parameters")
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
            self._caches_input.append(output)
            output = layer.forward(output)
            self._caches_output.append(output)
        return output

    def backward(self, grad_out):
        grad = grad_out
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    def summary(self, input_shape=None):
        if input_shape == None: 
            raise ValueError("Required input shape")

        random_sample = np.zeros(input_shape)
        self.forward(random_sample)
        
        console = Console(force_terminal=True)
        table = Table(title="Model Summary", title_style="bold_magenta")

        table.add_column("Layer", style="cyan", no_wrap=True)
        table.add_column("Input Shape", style='green')
        table.add_column("Output shape", style='green')
        
        table.add_column("Activation Function", style="black")
        table.add_column("Param #", style='yellow')

        activation_function_name = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'Softmax']

        for idx_layer in range(len(self._layers)):
            layer = self._layers[idx_layer]
            layer_name = type(layer).__name__ # get current layer name
            if layer_name in activation_function_name : 
                continue
            layer_input = self._caches_input[idx_layer].shape # get current layer input shape
            layer_output = self._caches_output[idx_layer].shape # get current layer output shape
            if idx_layer < len(self._layers)-1:
                layer_activation = type(self._layers[idx_layer + 1]).__name__ # get activation name
                layer_activation = layer_activation if layer_activation in activation_function_name else None

            # Get the number of parameters
            layer_params = 0
            for v in layer._parameters.values(): 
                layer_params += v.data.size

            # assign in the table 

            table.add_row(
                    f"[bold]{layer_name}[/bold]", 
                    f"[green]{layer_input}[/green]", 
                    f"[green]{layer_output}[/green]", 
                    f"[black]{layer_activation}[/black]", 
                    f"[yellow]{layer_params}[/yellow]", 
                    )

        console.print(table)
