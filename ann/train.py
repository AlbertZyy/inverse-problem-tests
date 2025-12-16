
import torch
import torch.nn as nn
from torch import Tensor, functional as F
import sucrose


class AnnEIT(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: list[int], output_dims: int):
        super().__init__()
        self.input_layer = nn.Linear(input_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dims)

    def forward(self, x: Tensor):
        x = torch.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        return torch.tanh(self.output_layer(x))


def main():
    pass