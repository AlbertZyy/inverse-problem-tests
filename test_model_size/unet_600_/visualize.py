
import sys
from typing import Callable

sys.path.append('./src')

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import MSELoss, Module
from tqdm import tqdm

from unet_600 import build_model

from dataset import NPZDataset

model_1, name_1 = build_model('cpu', 0.0, False)
model_2, name_2 = build_model('cpu', 0.0, True)
model_3, name_3 = build_model('cpu', 0.5, True)

model_1.eval()
model_2.eval()
model_3.eval()

validation_set = NPZDataset('./data/gdgn_64_64_validate/', 200)

NO_EVALUATE = True
NO_PLOT = False
save_dir = './test_model_size/plot_data/'


if not NO_EVALUATE:

    loader = DataLoader(validation_set, batch_size=20, shuffle=False)

    from common import loss_fn as cross_entropy
    mse = MSELoss()

    def validate(model: Module, loader: DataLoader,
                loss_fn: Callable[[Tensor, Tensor], Tensor]):
        model.eval()
        loss = 0
        count = 0

        for x, label in tqdm(loader, desc='Validation', unit='batch'):
            y_pred = model(x)
            loss += loss_fn(y_pred, label.flatten().to(dtype=torch.float32)).item()
            count += 1

        return loss / count


    for model, name in zip([model_1, model_2, model_3], [name_1, name_2, name_3]):
        cross_entropy_loss = validate(model, loader, cross_entropy)
        mse_loss = validate(model, loader, mse)
        print(f'Validation loss for {name}')
        print(f'  - cross entropy loss: {cross_entropy_loss}\n  - mse loss: {mse_loss}')

if not NO_PLOT:

    from matplotlib import pyplot as plt

    ID = [62, 92, 12, 22, 185, 176, 153, 118]

    for i in ID:
        fig = plt.figure(f"validate - {i}", figsize=(7.5, 7.5))
        data, label = validation_set[i]

        for k, (model, name) in enumerate(zip([model_1, model_2, model_3],
                                              [name_1, name_2, name_3])):
            pred = model(data[None, ...])
            axes = fig.add_subplot(2, 2, k+1)
            axes.imshow(pred.detach().reshape(64, 64))
            axes.invert_yaxis()
            axes.set_title(name)

        axes = fig.add_subplot(2, 2, 4)
        axes.imshow(label.to(dtype=torch.float32))
        axes.invert_yaxis()
        axes.set_title('label')
        fig.suptitle(f'validate - {i}')
        fig.savefig(f'{save_dir}vis_600_{i}.png')

    plt.show()
