
import sys
from typing import Callable, Tuple

import torch
from torch import Tensor, float32, device
import torch.nn as nn

sys.path.append("./src")

from fdm import LaplaceFDMSolver
from fractional import Fractional
from data_feature import DataFeatureFDMSolver


class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel: int, dtype=float32) -> None:
        super().__init__()
        in_, out_ = in_channel, out_channel
        self.conv_1 = nn.Conv2d(in_, out_, kernel, padding=kernel//2, dtype=dtype) # [N, 10, 64, 64]
        self.conv_2 = nn.Conv2d(out_, out_, kernel, padding=kernel//2, dtype=dtype) # [N, 10, 64, 64]
        self.bn = nn.BatchNorm2d(out_, momentum=0.9, dtype=dtype)
        self.down = nn.AvgPool2d(kernel_size=2) # [N, 10, 32, 32]

    def forward(self, phi: Tensor):
        phi = self.conv_2(self.conv_1(phi))
        out = self.down(torch.tanh_(self.bn(phi)))
        return out, phi

    __call__: Callable[[Tensor], Tuple[Tensor, Tensor]]


class ConvTBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel: int, dtype=float32) -> None:
        super().__init__()
        in_, out_ = in_channel, out_channel
        self.up = nn.ConvTranspose2d(in_, in_//2, 3, 2, 1, 1, dtype=dtype)
        self.convt_1 = nn.ConvTranspose2d(in_, out_, kernel, padding=kernel//2, dtype=dtype)
        self.convt_2 = nn.ConvTranspose2d(out_, out_, kernel, padding=kernel//2, dtype=dtype)
        self.bn = nn.BatchNorm2d(out_, momentum=0.9, dtype=dtype)

    def forward(self, phi: Tensor, conn: Tensor):
        phi = self.up(phi)
        phi = torch.cat([phi, conn], dim=1)
        del conn
        phi = self.convt_2(self.convt_1(phi))
        out = torch.tanh_(self.bn(phi))
        return out

    __call__: Callable[[Tensor, Tensor], Tensor]


class Unet(nn.Module):
    def __init__(self, n_channel: int, *, dtype=float32) -> None:
        super().__init__()

        self.cb1 = ConvBlock(n_channel, 12, 9, dtype=dtype) # [N, 12, 32, 32]
        self.cb2 = ConvBlock(12, 24, 5, dtype=dtype) # [N, 24, 16, 16]
        self.cb3 = ConvBlock(24, 48, 3, dtype=dtype) # [N, 48, 8, 8]
        self.cb4 = ConvBlock(48, 96, 3, dtype=dtype) # [N, 96, 4, 4]

        self.btm = nn.Conv2d(96, 192, 3, 1, 1, dtype=dtype)

        self.ctb4 = ConvTBlock(192, 96, 3, dtype=dtype) # [N, 96, 4, 4]
        self.ctb3 = ConvTBlock(96, 48, 3, dtype=dtype) # [N, 48, 16, 16]
        self.ctb2 = ConvTBlock(48, 24, 5, dtype=dtype) # [N, 24, 32, 32]
        self.ctb1 = ConvTBlock(24, 12, 9, dtype=dtype) # [N, 12, 64, 64]

        self.conv = nn.ConvTranspose2d(12, 1, 1, dtype=dtype)

    def forward(self, input: Tensor):

        phi, p1 = self.cb1(input)
        phi, p2 = self.cb2(phi)
        phi, p3 = self.cb3(phi)
        phi, p4 = self.cb4(phi)

        phi = self.btm(phi)

        phi = self.ctb4(phi, p4)
        del p4
        phi = self.ctb3(phi, p3)
        del p3
        phi = self.ctb2(phi, p2)
        del p2
        phi = self.ctb1(phi, p1)
        del p1
        phi = self.conv(phi)

        return phi

    __call__: Callable[[Tensor], Tensor]


class RevModel(nn.Module):
    def __init__(self, n_channel: int, lsolver: LaplaceFDMSolver, frac: Fractional,
                 *, network_dtype=float32) -> None:
        super().__init__()
        self.df_solver = DataFeatureFDMSolver(lsolver, frac) # [N, 16, 64, 64]
        self.bn = nn.BatchNorm2d(n_channel, momentum=0.9, dtype=lsolver.dtype)
        self.coordinate = lsolver.indexing.coordinate([-1, -1]) # [2, 64, 64]
        self.unet = Unet(n_channel+2, dtype=network_dtype)
        self.network_dtype = network_dtype


    def forward(self, input: Tensor):
        N = input.shape[0]
        coor = self.coordinate[None, ...].repeat(N, 1, 1, 1)
        phi = self.df_solver(input)
        del input
        phi = self.bn(phi)
        phi = torch.cat([phi, coor], dim=1)
        del coor
        phi = phi.to(self.network_dtype)
        phi = self.unet(phi)
        phi = torch.sigmoid_(phi.flatten())
        return phi

    __call__: Callable[[Tensor], Tensor]


def build_model(device: device, s: float=0.0, s_grad: bool=True):
    EXT = 63
    H = 2./EXT

    lsolver = LaplaceFDMSolver([EXT, EXT], [H, H], device=device)
    frac = Fractional(252, device=device)
    frac.from_npz(f"./data/laplace_beltrami_{EXT}_{EXT}.npz")
    frac.initialize(s)
    frac.s.requires_grad_(s_grad)

    model = RevModel(8, lsolver, frac, network_dtype=float32)
    model.to(device)

    NAME = "unet_100"

    if s_grad:
        FULL_NAME = NAME + f"_s{int(s*100)}"
    else:
        FULL_NAME = NAME + f"_s{int(s*100)}_no_grad"

    print(f"Model built: {FULL_NAME}, in device: {device}")

    n_p = sum(p.numel() for p in model.unet.parameters())
    print(f"Number of unet parameters: {n_p/1e6:.2f}M")

    try:
        model.load_state_dict(torch.load(f"./test_model_size/{NAME}_/checkpoints/{FULL_NAME}.pth", map_location=device))
        print(f"Checkpoint loaded.")
    except FileNotFoundError:
        print(f"No checkpoint found.")

    return model, FULL_NAME


if __name__ == "__main__":
    build_model('cpu', 0.0, False)
