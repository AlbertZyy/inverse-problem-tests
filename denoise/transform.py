
import sys
from typing import Dict
sys.path.append('./src')

import numpy as np
from numpy.typing import NDArray
import torch

from dataset import NPZDataset

TYPE = 'denoise'
CHANNEL_KEYS = ['1', '2', '3', '4', '5', '6', '8', '16']

if TYPE == 'noise':
    def noisy(datadict: Dict[str, NDArray]):
        g = np.random.randn()
        std = np.exp(g) * 1e-2
        for key, data in datadict.items():
            if key == 'label':
                continue
            noise = np.random.randn(*data.shape) * std
            datadict[key] += data * noise

    dataset = NPZDataset('./data/gdgn_64_64_validate/', num=200,
                         channel_keys=CHANNEL_KEYS, label_key='label')
    dataset.transform_to(noisy, './data/gdgn_64_64_validate_noisy/', tqdm=True)

elif TYPE == 'denoise':
    from cnn import build_model
    model, _ = build_model('cpu', '')
    dataset = NPZDataset('./data/gdgn_64_64_validate_noisy/', num=200,
                         channel_keys=CHANNEL_KEYS, label_key='label')

    def denoise(datadict: Dict[str, NDArray]):
        datalist = [datadict[key] for key in CHANNEL_KEYS]
        data = torch.from_numpy(np.stack(datalist, axis=0)) # [C, 2, NDof]
        CHANNEL, _, DOF = data.shape
        data = data.reshape(-1, DOF) # [C*2, NDof]
        denoised = model(data[None, ...]).detach().numpy()[0, ...].reshape(CHANNEL, 2, DOF) # [C, 2, NDof]
        for i, name in enumerate(CHANNEL_KEYS):
            datadict[name] = denoised[i, ...]

    dataset.transform_to(denoise, './data/gdgn_64_64_validate_denoised/', tqdm=True)
