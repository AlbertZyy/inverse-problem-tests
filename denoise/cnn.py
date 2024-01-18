
import torch
import torch.nn as nn
from torch import device, float64, float32


class NoiseExtractCNN(nn.Module):
    def __init__(self, n_channels: int, hidden: int):
        super(NoiseExtractCNN, self).__init__()
        NC = n_channels
        NHC = hidden
        self.conv1 = nn.Conv1d(NC, NHC, 3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv1d(NHC, NHC, 3, padding=1, padding_mode='circular')
        self.conv3 = nn.Conv1d(NHC, NHC, 3, padding=1, padding_mode='circular')
        self.conv4 = nn.Conv1d(NHC, NC, 3, padding=1, padding_mode='circular')

    def forward(self, input):
        x = torch.relu_(self.conv1(input.to(float32)))
        x = torch.relu_(self.conv2(x))
        x = torch.relu_(self.conv3(x))
        x = torch.relu_(self.conv4(x))
        return input - x.to(float64)


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
