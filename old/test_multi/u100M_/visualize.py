
import sys
from typing import Callable

sys.path.append('./src')

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import MSELoss, Module
from tqdm import tqdm

from unet_100 import build_model

from dataset import NPZDataset
from common import add_gaussian_noise


model_name_pairs = [
    build_model('cpu', 'P'),
    build_model('cpu', 'PN'),
]

validation_set = NPZDataset('./data/gdgn_64_64_4x2_validate/', 200)

NO_EVALUATE = False
NO_PLOT = False
save_dir = './test_multi/plot_data/'
if not NO_PLOT:
    import os
    os.makedirs(save_dir, exist_ok=True)


if not NO_EVALUATE:

    loader = DataLoader(validation_set, batch_size=40, shuffle=False)

    from common import loss_fn as cross_entropy
    mse = MSELoss()

    def validate(model: Module, loader: DataLoader,
                 loss_fn: Callable[[Tensor, Tensor], Tensor]):
        model.eval()
        loss = 0
        count = 0

        for x, label in tqdm(loader, desc='Validate', unit='batch'):
            add_gaussian_noise(x[:, :, 0, :], 0.05)
            y_pred = model(x)
            loss += loss_fn(y_pred, label.flatten().to(dtype=torch.float32)).item()
            count += 1

        return loss / count


    for model, name in model_name_pairs:
        cross_entropy_loss = validate(model, loader, cross_entropy)
        mse_loss = validate(model, loader, mse)
        print(f'Validation loss for {name}')
        print(f'  - cross entropy loss: {cross_entropy_loss}\n  - mse loss: {mse_loss}')


if not NO_PLOT:

    from matplotlib import pyplot as plt

    ID = [62, 92, 12, 22]

    for i in ID:
        fig = plt.figure(f"validate - {i}", figsize=(7.5, 2.5))
        data, label = validation_set[i]
        add_gaussian_noise(data[:, 0, :], 0.05)

        for k, (model, name) in enumerate(model_name_pairs[:2]):
            pred = model(data[None, ...])
            axes = fig.add_subplot(1, 3, k+1)
            axes.imshow(pred.detach().reshape(64, 64))
            axes.invert_yaxis()
            axes.set_title(name)

        axes = fig.add_subplot(1, 3, 3)
        axes.imshow(label.to(dtype=torch.float32))
        axes.invert_yaxis()
        axes.set_title('label')
        fig.suptitle(f'validate - {i}')
        fig.savefig(f'{save_dir}vis_u100M_P{i}.png')

    plt.show()
