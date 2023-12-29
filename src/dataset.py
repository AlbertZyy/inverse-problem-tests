
import numpy as np
import torch
from torch.utils.data import Dataset


class NPZDataset(Dataset):
    def __init__(self, folder: str, num: int, label_key='label') -> None:
        """
        @brief Initialize a dataset from `.npz` files.

        A single npz file is a sample, containing multiple channels and the label.
        (The array named `lebel_key` is the label, and others are channels.)

        @param folder: str. The path to the folder containing `.npz` files.
        @param num: int. The number of samples whose indices range from 0 to num-1.
        @param label_key: str. The name of the label data in a `.npz` file.
        """
        super().__init__()
        assert isinstance(folder, str)
        assert num > 0

        if folder[-1] != '/':
            folder += '/'

        self.path = folder
        self.num = num
        self.label_key = label_key

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        datadict = dict(np.load(self.path + f"{index}.npz"))
        label = datadict[self.label_key]
        del datadict[self.label_key]
        channels = [arr for arr in datadict.values()]
        data = np.stack(channels, axis=0)
        return torch.from_numpy(data), torch.from_numpy(label)
