
from itertools import chain
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD
from torch import Tensor
from torch.autograd import grad

import sucrose


class DenseNet(nn.Module):
    SIGMA_DICT = {
        "relu": torch.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "softplus": F.softplus,
        "softmax": F.softmax,
        "identity": lambda x: x,
        "none": lambda x: x,
        "relu6": F.relu6,
        "elu": F.elu,
        "sinc": torch.sinc
    }
    def __init__(self, input_dim: int, dims: list[int], sigma: list[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(dims)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, dims[i]))
                self.layers.append(DenseNet.SIGMA_DICT[sigma[i]])
            else:
                self.layers.append(nn.Linear(dims[i-1], dims[i]))
                self.layers.append(DenseNet.SIGMA_DICT[sigma[i]])

    def forward(self, data: Tensor) -> Tensor:
        for layer in self.layers:
            data = layer(data)
        return data


def operator_A(inputs: Tensor, u_net, gamma_net, phi_net, area: float):
    NN = inputs.shape[0]
    gamma = gamma_net(inputs)
    grad_u = grad(outputs=u_net(inputs), inputs=inputs)[0]
    grad_phi = grad(outputs=phi_net(inputs), inputs=inputs)[0]
    inner = torch.einsum("n, nd, nd -> ", gamma, grad_u, grad_phi)
    return inner / NN * area


def operator_B1(inputs: Tensor, u_net, ub, perimeter: float):
    return torch.mean(u_net(inputs) - ub, dim=0) * perimeter


def operator_B2(inputs: Tensor, normal: Tensor, u_net, gub, gammab, perimeter: float):
    grad_u = grad(outputs=u_net(inputs), inputs=inputs)[0]
    inner = torch.inner(grad_u, normal)
    return torch.mean(inner*gammab - gub, dim=0) * perimeter


def operator_B3(inputs: Tensor, gamma_net, gammab, perimeter: float):
    return torch.mean(gamma_net(inputs) - gammab, dim=0) * perimeter


def main(case: str):
    GD = 2
    DTYPE = torch.float64
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CONTEXT = {"dtype": DTYPE, "device": DEVICE}
    ssc = sucrose.scenario("iwan", case)

    u_net = ssc.partial(DenseNet, "u.")(input_dim=GD)
    gamma_net = ssc.partial(DenseNet, "gamma.")(input_dim=GD)
    phi1_net = ssc.partial(DenseNet, "phi.")(input_dim=GD)
    phi2_net = ssc.partial(DenseNet, "phi.")(input_dim=GD)

    sgd1 = ssc.partial(SGD, "sgd1")(chain(u_net.parameters(), gamma_net.parameters()))
    sgd2 = ssc.partial(SGD, "sgd2")(chain(phi1_net.parameters(), phi2_net.parameters()))
    Jn = ssc["optim.jn"]
    BETA = ssc["beta"]
    AREA = ssc["data.area"]
    PERIMETER = ssc["data.perimeter"]

    ssc.load_state_dict(
        u_net = u_net,
        gamma_net = gamma_net,
        phi1_net = phi1_net,
        phi2_net = phi2_net,
        sgd1 = sgd1,
        sgd2 = sgd2,
    )
    Nx, Ny = ssc["grid_size"]
    Xlin = torch.linspace(-1, 1, Nx, **CONTEXT)
    Ylin = torch.linspace(-1, 1, Ny, **CONTEXT)

    interior_pos = torch.stack(
        torch.meshgrid(Xlin, Ylin), dim=-1
    ).reshape(-1, GD) # (Nx*Ny, GD)
    boundary_pos = torch.concat([
        torch.stack([Xlin, torch.full(Nx, -1., **CONTEXT)], dim=-1),  # bottom
        torch.stack([Xlin, torch.full(Nx, 1., **CONTEXT)], dim=-1),   # top
        torch.stack([torch.full(Ny, -1., **CONTEXT), Ylin], dim=-1),  # left
        torch.stack([torch.full(Ny, 1., **CONTEXT), Ylin], dim=-1),   # right
    ])
    normal = torch.empty(((Nx + Ny)*2, GD), **CONTEXT)
    normal[:Nx, :] = torch.tensor([[0., -1.]], **CONTEXT)
    normal[Nx:2*Nx, :] = torch.tensor([[0., 1.]], **CONTEXT)
    normal[2*Nx:2*Nx+Ny, :] = torch.tensor([[-1., 0.]], **CONTEXT)
    normal[2*Nx+Ny:, :] = torch.tensor([[1., 0.]], **CONTEXT)

    for epoch in ssc.epoch_range(ssc["epochs"]):
        opA = operator_A(interior_pos, u_net, gamma_net, phi1_net, AREA)


if __name__ == "__main__":
    main("base")
