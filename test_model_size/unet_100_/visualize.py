
import sys
from typing import Callable

sys.path.append('./src')

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import MSELoss, Module

from unet_100 import build_model
from common import loss_fn as cross_entropy
from dataset import NPZDataset

model_1, name_1 = build_model('cpu', 0.0, False)
model_2, name_2 = build_model('cpu', 0.0, True)
model_3, name_3 = build_model('cpu', 0.5, True)

validation_set = NPZDataset('./data/gdgn_64_64_validate/', 200)
loader = DataLoader(validation_set, batch_size=20, shuffle=False)

mse = MSELoss()


def validate(model: Module, loader: DataLoader,
             loss_fn: Callable[[Tensor, Tensor], Tensor]):
    model.eval()
    loss = 0
    count = 0

    for x, label in loader:
        y_pred = model(x)
        loss += loss_fn(y_pred, label.flatten().to(dtype=torch.float32)).item()
        count += 1

    return loss / count


for model, name in zip([model_1, model_2, model_3], [name_1, name_2, name_3]):
    cross_entropy_loss = validate(model, loader, cross_entropy)
    mse_loss = validate(model, loader, mse)
    print(f'Validation loss for {name}')
    print(f'  - cross entropy loss: {cross_entropy_loss}\n  - mse loss: {mse_loss}')
