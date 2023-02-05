import torch
from typing import Callable
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        num_inputs = input_size
        for i in range(hidden_count):
            next_num_inputs = hidden_size
            self.layers += [nn.Linear(num_inputs, next_num_inputs)]
            num_inputs = next_num_inputs

        self.out = nn.Linear(hidden_size, num_classes)
        self.activation = activation

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out(x)
        return x
