
from typing import List, Optional, Tuple, Dict, Any, Callable, Iterator, Union

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import Dataset, BatchSampler, RandomSampler


_device = torch.device
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
    path: str
    num: int
    channel_keys: List[str]
    label_key: str

    def __init__(self, folder: str, *, num: int=-1,
                 channel_keys: Optional[List[str]]=None, label_key='label',
                 device: Union[_device, str, None]=None, pin_memory: Optional[bool]=False,
                 tqdm: bool=False) -> None:
        super().__init__()
        assert isinstance(folder, str)
        assert isinstance(num, int)
        kwargs = {'device': device, 'pin_memory': pin_memory}

        self.path = folder if folder[-1] == '/' else folder + '/'
        self.num = num

        if num == -1:
            import os
            num = len([f for f in os.listdir(folder) if f.endswith('.npz')])

        self.channel_keys = channel_keys if channel_keys else []
        self.label_key = label_key

        self._header_data_read = False

        iterator = range(self.num)
        if tqdm:
            from tqdm import tqdm as _tqdm
            iterator = _tqdm(iterator, desc=f"Loading", unit='sample')

        for index in iterator:

            if not self._header_data_read: # the first data
                pair = self._preload_data(0)
                data_shape = pair[0].shape
                data_dtype = pair[0].dtype
                label_shape = pair[1].shape
                label_dtype = pair[1].dtype
                self.data = torch.empty((self.num, *data_shape),
                                        dtype=data_dtype, **kwargs)
                self.labels = torch.empty((self.num, *label_shape),
                                          dtype=label_dtype, **kwargs)
                self._header_data_read = True

            else: # other data
                pair = self._preload_data(index)

            self.data[index].copy_(pair[0], non_blocking=True)
            self.labels[index].copy_(pair[1], non_blocking=True)

    def _preload_data(self, index: int):
        assert index >= 0 and index < self.num
        file_name = self.path + f"{index}.npz"
        datadict = dict(np.load(file_name))
        label = datadict[self.label_key]
        del datadict[self.label_key]
        if len(datadict) == 0:
            raise ValueError(f"No channel data found in the file '{file_name}'.")

        if len(self.channel_keys) == 0:
            self.channel_keys.extend(datadict.keys())
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
        return _TPZLoader(dataset=self, batch_size=batch_size, drop_last=drop_last)


class _TPZLoader():
    dataset: TPZDataset
    batch_size: int
    sampler: RandomSampler
    batch_sampler: BatchSampler
    _iterator: Iterator
    __initialized: bool = False

    def __init__(self, dataset: TPZDataset, batch_size: int, drop_last: bool=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = RandomSampler(self.dataset, num_samples=self.dataset.num)
        self.batch_sampler = BatchSampler(self.sampler, batch_size=batch_size, drop_last=drop_last)
        self._iterator = iter(self.batch_sampler)
        self.__initialized = True

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        indices = next(self._iterator)
        return self.dataset.__getitems__(indices)

    def __iter__(self):
        return self

    def __setattr__(self, attr: str, val: Any) -> None:
        if self.__initialized and attr in {
            'dataset', 'batch_size', 'sampler', 'batch_sampler', '_iterator',
        }:
            raise RuntimeError(f"Cannot set attribute {attr} after {self.__class__.__name__} is initialized.")

        super().__setattr__(self, attr, val)


if __name__ == '__main__':
    from time import time

    dataset = TPZDataset("./data/gdgn_64_64_train/",
                         num=50,
                         device='cpu',
                         pin_memory=False,
                         tqdm=True)

    print(dataset.channel_keys)

    t1 = time()

    for data, label in dataset.loader(10, drop_last=False):
        print(data.stride())

    print(time() - t1)
