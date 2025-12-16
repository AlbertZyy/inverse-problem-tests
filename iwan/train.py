
import math
from itertools import chain
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD
from torch import Tensor
from torch.autograd import grad

import sucrose

sucrose.logger.setLevel("INFO")
DTYPE = torch.float32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONTEXT = {"dtype": DTYPE, "device": DEVICE}
I_CONTEXT = {"dtype": torch.int32, "device": DEVICE}


def check_tensor(x, name):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} contains NaN or INF")


class StableSinc(nn.Module):
    def forward(self, x: Tensor):
        output = torch.empty_like(x)
        flag = torch.abs(x) < 1e-2
        pix = torch.pi * x
        x2 = pix * pix
        # 二阶泰勒：sinc(x) ≈ 1 - (πx)^2 / 6
        taylor = 1.0 - x2[flag] / 6.0
        regular = torch.sin(pix[~flag]) / pix[~flag]
        output = output.index_put([flag], taylor)
        output = output.index_put([~flag], regular)
        return output


class DenseNet(nn.Module):
    SIGMA_DICT = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "none": nn.Identity,
        "elu": nn.ELU,
        "sinc": StableSinc
    }
    def __init__(self, input_dim: int, dims: list[int], sigma: list[str]) -> None:
        super().__init__()
        self.layers = nn.Sequential()

        for i in range(len(dims)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, dims[i]))
                self.layers.append(DenseNet.SIGMA_DICT[sigma[i]]())
            else:
                self.layers.append(nn.Linear(dims[i-1], dims[i]))
                self.layers.append(DenseNet.SIGMA_DICT[sigma[i]]())

    def forward(self, data: Tensor) -> Tensor:
        return self.layers(data)


def norm_A(inputs: Tensor, u_net, gamma_net, phi_nets: list, area: float) -> Tensor:
    NN = inputs.shape[0]
    GRAD_OUTPUTS = torch.ones((NN, 1), **CONTEXT)
    gamma = gamma_net(inputs)
    check_tensor(gamma, "gamma(int)")
    grad_u = grad(outputs=u_net(inputs), inputs=inputs, grad_outputs=GRAD_OUTPUTS, create_graph=True)[0]
    check_tensor(grad_u, "grad_u(int)")
    grad_phis = grad(
        outputs=[net(inputs) for net in phi_nets],
        inputs=inputs,
        grad_outputs=[GRAD_OUTPUTS]*len(phi_nets),
        create_graph=True
    )
    result = 0.0
    for grad_phi in grad_phis:
        check_tensor(grad_phi, "grad_phi(int)")
        inner = torch.einsum("nf, nd, nd -> ", gamma, grad_u, grad_phi)
        result = result + (inner / NN * area)**2
    return result

def norm_B1(inputs: Tensor, u_net, ub, perimeter: float):
    val = u_net(inputs)
    check_tensor(val, "u(bdry)")
    return F.mse_loss(val, ub) * perimeter

def norm_B2(inputs: Tensor, normal: Tensor, bdry_edge: Tensor, u_net, gub, gammab, perimeter: float):
    NN = inputs.shape[0]
    GRAD_OUTPUTS = torch.ones((NN, 1), **CONTEXT)
    grad_u = grad(outputs=u_net(inputs), inputs=inputs, grad_outputs=GRAD_OUTPUTS, create_graph=True)[0]
    check_tensor(grad_u, "grad_u(bdry)")
    inner = torch.einsum("nd, nd -> n", grad_u, normal) # (NE,)
    val = torch.zeros(NN, **CONTEXT).requires_grad_() # (NN,)
    for i in range(2):
        val = val.index_add(0, index=bdry_edge[:, i], source=inner)
    return F.mse_loss(val[:, None]*gammab, gub) * perimeter

def norm_B3(inputs: Tensor, gamma_net, gammab, perimeter: float):
    val = gamma_net(inputs) - gammab
    check_tensor(val, "gamma(bdry)")
    ZERO = torch.zeros_like(val).requires_grad_(False)
    return F.mse_loss(val, ZERO) * perimeter

def zero_grad(optims):
    for optim in optims:
        optim.zero_grad()


def main(case: str):
    GD = 2
    ssc = sucrose.scenario("iwan", case)

    u_net = ssc.partial(DenseNet, "u.")(input_dim=GD)
    gamma_net = ssc.partial(DenseNet, "gamma.")(input_dim=GD)
    phi1_net = ssc.partial(DenseNet, "phi.")(input_dim=GD)
    phi2_net = ssc.partial(DenseNet, "phi.")(input_dim=GD)

    sgd_u = ssc.partial(SGD, "sgd1")(u_net.parameters())
    sgd_gamma = ssc.partial(SGD, "sgd1")(gamma_net.parameters())
    sgd_phi1 = ssc.partial(SGD, "sgd2")(phi1_net.parameters(), maximize=True)
    sgd_phi2 = ssc.partial(SGD, "sgd2")(phi2_net.parameters(), maximize=True)

    writter = ssc.start_pytorch_tensorboard()

    bdry_info = np.load(ssc["data.bdry_info_path"])
    node = torch.from_numpy(bdry_info["node"]).to(**CONTEXT)
    node.requires_grad_()
    isBDNode = torch.from_numpy(bdry_info["is_bdry_node"]).to(dtype=torch.bool, device=DEVICE)
    bdry_node = torch.clone(node[isBDNode]).requires_grad_()
    bdry_edge = torch.from_numpy(bdry_info["bdry_edge"]).to(**I_CONTEXT)
    bdry_edge_normal = torch.from_numpy(bdry_info["bdry_edge_normal"]).to(**CONTEXT)
    bdry_edge_barycenter = torch.from_numpy(bdry_info["bdry_edge_barycenter"]).to(**CONTEXT)
    bdry_edge_barycenter.requires_grad_()
    gd = torch.from_numpy(
        np.load(ssc["data.gd_path"])[0, :, None] # (252, 1)
    ).to(**CONTEXT)
    gn = torch.from_numpy(
        np.load(ssc["data.gn_path"])[0, :, None] # (252, 1)
    ).to(**CONTEXT)
    label = torch.from_numpy(np.load(ssc["data.label_path"])['label'])
    label = label.to(**CONTEXT) * 9 + 1.0
    gammab = 1.0

    Jn = ssc["optim.jn"]
    COEF_A = ssc["loss.coef_A"]
    COEF_AG = ssc["loss.coef_AG"]
    COEF_B1 = ssc["loss.coef_B1"]
    COEF_B2 = ssc["loss.coef_B2"]
    COEF_B3 = ssc["loss.coef_B3"]
    AREA = ssc["data.area"]
    PERIMETER = ssc["data.perimeter"]
    B = ssc["phi_constraint"]

    ssc.load_state_dict(
        u_net = u_net,
        gamma_net = gamma_net,
        phi1_net = phi1_net,
        phi2_net = phi2_net,
        sgd_u = sgd_u,
        sgd_gamma = sgd_gamma,
        sgd_phi1 = sgd_phi1,
        sgd_phi2 = sgd_phi2,
    )

    for epoch in ssc.epoch_range(ssc["epochs"]):
        random_node = torch.rand_like(node) * 2 - 1.
        random_node.requires_grad_()
        # update u
        normA = norm_A(random_node, u_net, gamma_net, [phi1_net, phi2_net], AREA)
        normB1 = norm_B1(bdry_node, u_net, gd, PERIMETER)
        normB2 = norm_B2(bdry_edge_barycenter, bdry_edge_normal, bdry_edge,
                         u_net, gn, gammab, PERIMETER)
        loss = COEF_A*normA + COEF_B1*normB1 + COEF_B2*normB2 # u_net 与 normB3 无关
        zero_grad([sgd_u, sgd_gamma, sgd_phi1, sgd_phi2])
        loss.backward(retain_graph=True)
        sgd_u.step()

        # update phi1
        for _ in range(Jn):
            normA = norm_A(random_node, u_net, gamma_net, [phi1_net, phi2_net], AREA)
            zero_grad([sgd_u, sgd_gamma, sgd_phi1, sgd_phi2])
            normA.backward(retain_graph=True)
            sgd_phi1.step()

        # update gamma
        normA = norm_A(random_node, u_net, gamma_net, [phi1_net, phi2_net], AREA)
        normB3 = norm_B3(bdry_node, gamma_net, gammab, PERIMETER)
        loss = COEF_AG*normA + COEF_B3*normB3 # gamma 与 normB1，normB2 无关
        zero_grad([sgd_u, sgd_gamma, sgd_phi1, sgd_phi2])
        loss.backward(retain_graph=True)
        sgd_gamma.step()

        # update phi2
        for _ in range(Jn):
            normA = norm_A(random_node, u_net, gamma_net, [phi1_net, phi2_net], AREA)
            zero_grad([sgd_u, sgd_gamma, sgd_phi1, sgd_phi2])
            normA.backward(retain_graph=True)
            sgd_phi2.step()

        with torch.no_grad():
            for phi_net in [phi1_net, phi2_net]:
                upper = math.sqrt(2.0*B)
                for param in phi_net.parameters():
                    param.clamp_(-upper, upper)

        writter.add_scalar("normA(train)", normA.item(), ssc.num_steps)
        writter.add_scalar("normB1(train)", normB1.item(), ssc.num_steps)
        writter.add_scalar("normB2(train)", normB2.item(), ssc.num_steps)
        writter.add_scalar("normB3(train)", normB3.item(), ssc.num_steps)
        # writter.add_scalar("loss(train)", loss.item(), ssc.num_steps)
        if ssc.num_steps % 200 == 0:
            img = gamma_net(node).reshape(1, 64, 64)
            # color map
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            writter.add_image("gamma(train)", img, ssc.num_steps)
        ssc.step()

        ssc.save_state_dict(
            500, # interval
            u_net = u_net,
            gamma_net = gamma_net,
            phi1_net = phi1_net,
            phi2_net = phi2_net,
            sgd_u = sgd_u,
            sgd_gamma = sgd_gamma,
            sgd_phi1 = sgd_phi1,
            sgd_phi2 = sgd_phi2,
        )


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main("base/test4")
