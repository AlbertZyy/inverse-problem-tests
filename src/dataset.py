
import os
from typing import (
    Optional, Union,
    List, Tuple, Dict, Collection, Iterable, Sequence,
    Any, Callable, Iterator,
)

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import Dataset, BatchSampler, RandomSampler, DataLoader


_device = torch.device
ArrayFunction = Callable[[NDArray[Any]], NDArray[Any]]


class NPZDataset(Dataset):
    path: str
    names_seq: Sequence
    channel_keys: List[str]
    label_key: str
    keep_dim: bool
    _cache: Optional[List[Optional[Tuple[Tensor, Tensor]]]]

    def __init__(self, folder: str, names: Union[Sequence[Any], int]=-1, *,
                 channel_keys: Optional[List[str]]=None, label_key='label',
                 keep_dim=False,
                 use_cache=False) -> None:
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

        self.path = os.path.join(folder, '')

        if isinstance(names, int):
            if names == -1:
                self.names_seq = [f for f in os.listdir(folder) if f.endswith('.npz')]
            else:
                self.names_seq = range(names)
        else:
            # NOTE: The number of files must be limited, so the type of `names`
            # provided here must be Sequence instead of Iterable.
            self.names_seq = names

        NUM = len(self.names_seq)
        self.channel_keys = list(channel_keys) if (channel_keys is not None) else []
        self.label_key = label_key
        self.keep_dim = keep_dim
        self._cache = [None, ] * NUM if use_cache else None

    def __len__(self) -> int:
        return len(self.names_seq)

    def has_cache(self, index: int):
        if self._cache is None:
            return False
        return self._cache[index] is not None

    def _read_data(self, fname: str):
        file_name = os.path.join(self.path, f"{fname}.npz")
        with np.load(file_name) as f:
            datadict = dict(f)
        label = datadict[self.label_key]
        del datadict[self.label_key]
        if len(datadict) == 0:
            raise ValueError(f"No channel data found in the file '{file_name}'.")

        if len(self.channel_keys) == 0:
            self.channel_keys.extend(datadict.keys())
        channels = [datadict[key] for key in self.channel_keys]

        if not self.keep_dim and len(channels) == 1:
            data = channels[0]
        else:
            data = np.stack(channels, axis=0)
        pair = torch.from_numpy(data), torch.from_numpy(label)
        return pair

    def _read_batch(self, names: Sequence[str]):
        return [self._read_data(name) for name in names]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.has_cache(index):
            return self._cache[index]
        else:
            pair = self._read_data(self.names_seq[index])

            if self._cache is not None:
                self._cache[index] = pair
            return pair

    def transform_to(self, func: Callable[[Dict[str, NDArray]], None],
                     destination_folder: str, *, tqdm=False):
        """
        @brief Transform the data and save to a new folder.
        """
        os.path.join(destination_folder, '')
        os.makedirs(destination_folder, exist_ok=True)

        if tqdm:
            from tqdm import tqdm
            iterator = tqdm(self.names_seq, desc=f"Transform", unit='sample')
        else:
            iterator = self.names_seq

        for fname in iterator:
            pathname_from = os.path.join(self.path, f"{fname}.npz")
            pathname_to = os.path.join(destination_folder, f"{fname}.npz")
            datadict = dict(np.load(pathname_from))
            func(datadict)
            np.savez(pathname_to, **datadict)


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


class TPZDataset(NPZDataset):
    def __init__(self, folder: str, names: Union[Sequence[Any], int]=-1, *,
                 channel_keys: Optional[Iterable[str]]=None, label_key='label',
                 keep_dim: bool=False, num_workers: int=0,
                 device: Union[_device, str, None]=None, pin_memory: Optional[bool]=False,
                 tqdm: bool=False) -> None:
        """
        @brief Initialize a dataset from `.npz` files.

        @param folder: str. The path to the folder containing `.npz` files.
        @param names: Collection | int. A collection of file name strings (excluding file extension).
               If an positive integer is given, filenames will be strings of integers starting from '0'.
               When -1 is given, take all `.npz` files in the folder.
               Defaults to -1.
        @param channel_keys: Iterable[str], optional. The names of the channel data in each `.npz` file.
               If None, all channels appearing in the first file are used as the channels of the data set.
               Defaults to None.
        @param label_key: str, optional. The name of the label data in a `.npz`.
               Defaults to 'label'.
        @param keep_dim: bool, optional. Whether to build an extra channel axis/dim\
               even if there is only one key-value pair besides the label. Defaults to False.
        @param device: torch.device, optional.
        @param pin_memory: bool | str, optional.
        @param tqdm: bool, optional. Whether to show a progress bar, defaults to False.
        """
        super().__init__(folder=folder, names=names, channel_keys=channel_keys,
                         label_key=label_key, keep_dim=keep_dim, use_cache=False)
        kwargs = {'device': device, 'pin_memory': pin_memory}
        NUM = len(self.names_seq)
        if num_workers == 0:
            worker_batch_size = NUM // 4
        else:
            worker_batch_size = NUM // num_workers // 4
        if worker_batch_size == 0:
            worker_batch_size = 1
        _preloader = DataLoader(dataset=self.names_seq,
                                batch_size=worker_batch_size,
                                num_workers=num_workers,
                                collate_fn=self._read_batch)
        self._header_data_read = False

        iterator = range(len(_preloader))
        if tqdm:
            from tqdm import tqdm as _tqdm
            iterator = _tqdm(iterator, desc=f"Loading", unit=f'x{worker_batch_size}sample')

        for batch_idx, pairs in zip(iterator, _preloader):
            for local_idx, pair in enumerate(pairs):
                index = batch_idx * worker_batch_size + local_idx

                if not self._header_data_read: # the first data
                    data_shape = pair[0].shape
                    data_dtype = pair[0].dtype
                    label_shape = pair[1].shape
                    label_dtype = pair[1].dtype
                    self.data = torch.empty((NUM, *data_shape),
                                            dtype=data_dtype, **kwargs)
                    self.labels = torch.empty((NUM, *label_shape),
                                            dtype=label_dtype, **kwargs)
                    self._header_data_read = True

                self.data[index].copy_(pair[0], non_blocking=True)
                self.labels[index].copy_(pair[1], non_blocking=True)

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
        self.sampler = RandomSampler(self.dataset, num_samples=len(self.dataset))
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

        super().__setattr__(attr, val)


if __name__ == '__main__':
    from time import time

    dataset = TPZDataset("./data/gdgn_64_64_train/", range(0, 100, 1),
                         num_workers=2,
                         device='cpu',
                         pin_memory=False,
                         tqdm=True)

    print(dataset.path)
    print(list(dataset.names_seq))

    t1 = time()

    for data, label in dataset.loader(10, drop_last=False):
        print(data.shape, label.shape)

    print(time() - t1)
