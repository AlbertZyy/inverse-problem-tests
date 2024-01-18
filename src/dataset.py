
from typing import List, Optional, Tuple, Dict, Any, Callable

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import Dataset


ArrayFunction = Callable[[NDArray[Any]], NDArray[Any]]


class NPZDataset(Dataset):
    def __init__(self, folder: str, num: int=-1, label_key='label', use_cache=False) -> None:
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

        if num == -1:
            import os
            num = len([f for f in os.listdir(folder) if f.endswith('.npz')])

        self.label_key = label_key
        self.use_cache = use_cache
        self._cache: List[Optional[Tuple[Tensor, Tensor]]] = [None, ] * num

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.use_cache and self._cache[index] is not None:
            return self._cache[index]
        else:
            datadict = dict(np.load(self.path + f"{index}.npz"))
            label = datadict[self.label_key]
            del datadict[self.label_key]
            channels = [arr for arr in datadict.values()]
            data = np.stack(channels, axis=0)
            pair = torch.from_numpy(data), torch.from_numpy(label)
            if self.use_cache:
                self._cache[index] = pair
            return pair

    def transform_to(self, func: ArrayFunction, destination_folder: str):
        """
        @brief Transform the data and save to a new folder.
        """
        if destination_folder[-1] != '/':
            destination_folder += '/'

        import os
        os.makedirs(destination_folder, exist_ok=True)

        for i in range(self.num):
            datadict = dict(np.load(self.path + f"{i}.npz"))
            for key, value in datadict.items():
                if key == self.label_key:
                    continue
                datadict[key] = func(value)
            np.savez(destination_folder + f"{i}.npz", **datadict)


def load_npz_dataset(config_dict: Dict[str, Any]) -> NPZDataset:
    """
    @brief Setup a npz dataset from a config dict.

    The dict may contain the following keys:
    - location: str. The path to the folder containing `.npz` files.
    - num: int, optional. The number of samples whose indices range from 0 to num-1,\
           defaults to -1.
    - label_key: str, optional. The name of the label data in a `.npz` file, defaults to "label".
    - use_cache: bool, optional. Whether to cache the data, defaults to False.
    """
    location: str   = config_dict['location']
    num: int        = config_dict.get('num', -1)
    label_key: str  = config_dict.get('label_key', "label")
    use_cache: bool = config_dict.get('use_cache', False)

    if location[-1] != '/':
        location += '/'

    return NPZDataset(location, num, label_key, use_cache)
