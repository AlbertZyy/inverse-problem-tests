
import torch
import torch.nn as nn
from torch import device, float64


class NoiseExtractCNN(nn.Module):
    def __init__(self, n_channels: int, hidden: int):
        super(NoiseExtractCNN, self).__init__()
        NC = n_channels
        NHC = hidden
        kwargs = dict(padding_mode='circular', dtype=float64)
        self.conv1 = nn.Conv1d(NC, NHC, 9, padding=4, **kwargs)
        self.conv2 = nn.Conv1d(NHC, NHC, 3, padding=1, **kwargs)
        self.bn12 = nn.BatchNorm1d(NHC, momentum=0.9, dtype=float64)
        self.down = nn.AvgPool1d(kernel_size=2)

        self.btm = nn.Conv1d(NHC, NHC, 3, padding=1, **kwargs)

        self.up = nn.ConvTranspose1d(NHC, NHC, 3, 2, 1, 1, dtype=float64)
        self.conv3 = nn.Conv1d(NHC, NHC, 3, padding=1, **kwargs)
        self.conv4 = nn.Conv1d(NHC, NC, 9, padding=4, **kwargs)
        self.bn34 = nn.BatchNorm1d(NC, momentum=0.9, dtype=float64)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = torch.relu_(self.bn12(x))
        x = self.down(x)

        x = self.btm(x)

        x = self.up(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.relu_(self.bn34(x))
        return input + x


def build_model(device: device, tag: str):

    model = NoiseExtractCNN(8*2, 128)
    model.to(device)

    NAME = "cnn"

    FULL_NAME = (NAME + '_' + tag) if tag else NAME

    print(f"Model built: {FULL_NAME}, in device: {device}")

    n_p = sum(p.numel() for p in model.parameters())
    print(f"Number of unet parameters: {n_p/1e6:.2f}M")

    try:
        model.load_state_dict(torch.load(f"./denoise/checkpoints/{FULL_NAME}.pth", map_location=device))
        print(f"Checkpoint loaded.")
    except FileNotFoundError:
        print(f"No checkpoint found.")

    return model, FULL_NAME


if __name__ == "__main__":
    build_model('cpu', '')
