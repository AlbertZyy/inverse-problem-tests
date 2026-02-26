
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import grad


def ccw_index(edge_length: int, device=None):
    NN = edge_length
    assert NN % 4 == 0
    EM = NN // 4
    original = torch.arange(NN, dtype=torch.int32, device=device)
    bottom_edge = original[EM+1:3*EM:2]
    right_edge = original[3*EM:4*EM]
    top_edge = torch.flip(original[EM:3*EM-1:2], dims=[0])
    left_edge = torch.flip(original[0:EM], dims=[0])
    return torch.concat([bottom_edge, right_edge, top_edge, left_edge], dim=0)


class AnnEIT(nn.Module):
    ACTIVATE_MAP = {
        "relu": torch.relu,
        "tanh": torch.tanh,
        "softplus": F.softplus,
        "leaky_relu": F.leaky_relu
    }
    def __init__(self, input_dims: int, hidden_dims: list[int], output_dims: int, activation: str = "leaky_relu"):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dims, elementwise_affine=False)
        self.input_layer = nn.Linear(input_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dims)
        self.activate = AnnEIT.ACTIVATE_MAP[activation]

    def forward(self, x: Tensor):
        x = self.layer_norm(x)
        x = self.activate(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activate(layer(x))

        return self.output_layer(x)


class PINNEIT(nn.Module):
    def __init__(self, channel: int, bdry_nodes: int, all_nodes: int, hidden_dims: int):
        super().__init__()
        HD = hidden_dims
        self.channel = channel
        self.bdry_nodes = bdry_nodes
        self.all_nodes = all_nodes
        self.conv = nn.Conv1d(channel, channel, kernel_size=3, padding=1, padding_mode="circular")
        self.bn = nn.BatchNorm1d(channel)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=3, padding=1, padding_mode="circular")
        self.bn2 = nn.BatchNorm1d(channel)
        self.sigma_branch = AnnEIT(bdry_nodes*channel, [HD, HD, HD], all_nodes)
        self.bdry_potential_branch = AnnEIT(bdry_nodes*channel, [HD, HD], bdry_nodes*channel)
        self.all_potential_branch = AnnEIT(bdry_nodes*channel, [HD, HD], all_nodes*channel)

    def forward(self, x: Tensor): # (N, channel, bdry nodes)
        x = self.bn(self.conv(x))
        x = F.leaky_relu(x)
        x = self.bn2(self.conv2(x)) # (N, channel, bdry nodes)
        x = F.leaky_relu(x)
        backbone = x.reshape(x.shape[0], -1) # (N, channel*bdry nodes)
        sigma = self.sigma_branch(backbone)
        bdry_potential = self.bdry_potential_branch(backbone).reshape(x.shape[0], self.channel, self.bdry_nodes) # (N, bdry nodes)
        all_potential = self.all_potential_branch(backbone).reshape(x.shape[0], self.channel, self.all_nodes) # (N, all nodes)
        return sigma, bdry_potential, all_potential


def mean_norm1_error(pred: Tensor, true: Tensor):
    return torch.mean(torch.abs(pred - true))

def mean_norm1(data: Tensor):
    return torch.mean(torch.abs(data))


class MixedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.betas_initial = nn.Buffer(
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        )
        self.betas = nn.Parameter(
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        )

    def forward(
        self,
        inputs: Tensor,
        sigma_pred: Tensor,
        sigma_true: Tensor,
        bdry_potential_pred: Tensor,
        bdry_potential_true: Tensor,
        all_potential_pred: Tensor,
        all_potential_true: Tensor
    ):
        GRAD_OUTPUTS = torch.ones_like(inputs)
        sigma_bce = F.binary_cross_entropy_with_logits(sigma_pred, sigma_true)
        bdry_potential_n1 = mean_norm1_error(bdry_potential_pred, bdry_potential_true)
        all_potential_n1 = mean_norm1_error(all_potential_pred, all_potential_true)

        grad_bdry_potential = grad(
            outputs=bdry_potential_pred,
            inputs=inputs,
            grad_outputs=GRAD_OUTPUTS,
            create_graph=True
        )
        compatibility_loss = mean_norm1(grad_bdry_potential[0])
        betas = torch.exp(self.betas)

        return (
            sigma_bce,
            betas[0] * bdry_potential_n1,
            betas[1] * all_potential_n1,
            betas[2] * compatibility_loss,
            mean_norm1_error(self.betas, self.betas_initial)
        )
