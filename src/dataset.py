
from typing import List, Optional, Tuple, Dict, Any, Callable

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, dtype, device
from torch.utils.data import Dataset, BatchSampler, RandomSampler


ArrayFunction = Callable[[NDArray[Any]], NDArray[Any]]


class NPZDataset(Dataset):
    def __init__(self, folder: str, num: int=-1, channel_keys: Optional[List[str]]=None,
                 label_key='label', *, use_cache=False) -> None:
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

        self.channel_keys = channel_keys
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

            if self.channel_keys is None:
                channels = [arr for arr in datadict.values()]
            else:
                channels = [datadict[key] for key in self.channel_keys]

            data = np.stack(channels, axis=0)
            pair = torch.from_numpy(data), torch.from_numpy(label)

            if self.use_cache:
                self._cache[index] = pair
            return pair

    def transform_to(self, func: Callable[[Dict[str, NDArray]], None],
                     destination_folder: str, *, tqdm=False):
        """
        @brief Transform the data and save to a new folder.
        """
        if destination_folder[-1] != '/':
            destination_folder += '/'

        import os
        os.makedirs(destination_folder, exist_ok=True)

        if tqdm:
            from tqdm import tqdm
            iterator = tqdm(range(self.num), desc=f"Transform", unit='sample')
        else:
            iterator = range(self.num)

        for i in iterator:
            datadict = dict(np.load(self.path + f"{i}.npz"))
            func(datadict)
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


class TPZDataset(Dataset):
    def __init__(self, folder: str,
                 channel_shape: Tuple[int, ...],
                 label_shape: Tuple[int, ...],
                 channel_dtype: dtype,
                 label_dtype: dtype,
                 *,
                 num: int=-1,
                 channel_keys: Optional[List[str]]=None,
                 label_key='label',
                 device: device=None,
                 pin_memory: bool=False,
                 tqdm: bool=False) -> None:
        super().__init__()
        assert isinstance(folder, str)
        assert num > 0
        kwargs = {'device': device, 'pin_memory': pin_memory}

        if folder[-1] != '/':
            folder += '/'

        self.path = folder
        self.num = num

        if num == -1:
            import os
            num = len([f for f in os.listdir(folder) if f.endswith('.npz')])

        self.channel_keys = channel_keys
        self.label_key = label_key
        CHANNEL = len(channel_keys)
        self.data = torch.empty((num, CHANNEL, *channel_shape), dtype=channel_dtype, **kwargs)
        self.labels = torch.empty((num, *label_shape), dtype=label_dtype, **kwargs)

        if tqdm:
            from tqdm import trange
            iterator = trange(0, self.num, desc=f"Loading", unit='sample')
        else:
            iterator = range(self.num)
        for index in iterator:
            pair = self._preload_data(index)
            self.data[index], self.labels[index] = pair

    def _preload_data(self, index: int):
        assert index >= 0 and index < self.num
        datadict = dict(np.load(self.path + f"{index}.npz"))
        label = datadict[self.label_key]
        del datadict[self.label_key]

        if self.channel_keys is None:
            channels = [arr for arr in datadict.values()]
        else:
            channels = [datadict[key] for key in self.channel_keys]

        data = np.stack(channels, axis=0)
        pair = torch.from_numpy(data), torch.from_numpy(label)
        return pair

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.data[index].contiguous(), self.labels[index].contiguous()

    def __getitems__(self, indices) -> Tuple[Tensor, Tensor]:
        return self.data[indices].contiguous(), self.labels[indices].contiguous()

    def loader(self, batch_size: int, drop_last=False):
        return _TPZLoader(self, batch_size, drop_last)


class _TPZLoader():
    def __init__(self, dataset: TPZDataset, batch_size: int, drop_last: bool=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = RandomSampler(self.dataset, num_samples=self.dataset.num)
        self.batch_sampler = BatchSampler(self.sampler, batch_size=batch_size, drop_last=drop_last)
        self.iterator = iter(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self.iterator)
        return self.dataset.__getitems__(indices)

    def __iter__(self):
        return self


if __name__ == '__main__':
    from time import time

    dataset = TPZDataset("./data/gdgn_64_64_train/",
                         channel_shape=(2, 252),
                         label_shape=(64, 64),
                         channel_dtype=torch.float64,
                         label_dtype=torch.bool,
                         num=50,
                         channel_keys=['1', '2', '3', '4', '5', '6', '8', '16'],
                         label_key='label',
                         device='cpu',
                         pin_memory=False)

    t1 = time()

    for data, label in dataset.loader(10, drop_last=False):
        pass

    print(time() - t1)
