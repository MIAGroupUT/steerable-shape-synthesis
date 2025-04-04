"""Various INR models.

Mostly taken from:
https://github.com/MIAGroupUT/SIRE-segmentation/blob/main/src/sire/inr/models.py
"""

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, layers=[3, 256, 256, 256, 256, 256, 1]):
        """Initialize the network.

        Layers is a list of ints, cointaining the number of features per
        layer."""
        super().__init__()

        self.n_layers = len(layers) - 1

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, coords):
        """The forward function of the network."""
        x = coords
        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            z = layer(x)
            x = torch.relu(z)

        # Propagate through final layer and return the output
        x = self.layers[-1](x)
        return x, coords


class Siren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, layers=[3, 256, 256, 256, 1], weight_init=True, omega=30):
        """Initialize the network."""

        super().__init__()

        self.n_layers = len(layers) - 1
        self.omega = omega

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    else:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, coords):
        """The forward function of the network."""
        x = coords
        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            z = layer(x)
            x = torch.sin(self.omega * z)

        # Propagate through final layer and return the output
        x = self.layers[-1](x)
        return x, coords


class ParallelMLP(nn.Module):
    def __init__(self, n_mlps=5, layers=[3, 256, 256, 256, 1]):
        """Train parallel MLPs with shared input and a single output node"""
        super().__init__()
        self.n_mlps = n_mlps
        self.models = []
        for i in range(self.n_mlps):
            self.models.append(MLP(layers=layers))

        self.models = nn.Sequential(*self.models)

    def forward(self, coords):
        x = coords
        outputs = []
        for model in self.models:
            output, coords = model(x)
            outputs.append(output)

        return torch.cat(outputs, dim=-1), None


class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), requires_grad=True)
        )
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max()  # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(
            input, self.weight * scale.unsqueeze(1), self.bias
        )


class LipschitzMLP(torch.nn.Module):
    def __init__(self, layers=[3, 256, 256, 256, 1]):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for ii in range(len(layers) - 2):
            self.layers.append(LipschitzLinear(layers[ii], layers[ii + 1]))

        self.layer_output = LipschitzLinear(layers[-2], layers[-1])
        self.relu = torch.nn.ReLU()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc * self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, coords):
        x = coords
        x[..., :3] = x[..., :3] * 100  # See the paper A.1

        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)

        return self.layer_output(x), coords
