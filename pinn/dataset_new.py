
import os
from typing import Any, Tuple, List, Sequence, Optional, Union, TypeVar, Callable

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor



class NPZDataset(Dataset):
    path: str
    names_seq: Sequence
    channel_keys: List[str]
    label_key: str
    keep_dim: bool
    _cache: list[dict[str, Tensor] | None] | None

    def __init__(
            self,
            folder: str,
            names: Union[Sequence[Any], int] = -1, *,
            use_cache = False
        ) -> None:
        """Initialize a dataset from `.npz` files.
        A single file is a sample, containing multiple channels and the label.
        (The array named `lebel_key` is the label, and others are channels.)

        Args:
            folder (str): The path to the folder containing `.npz` files.
            num (int): The number of samples whose indices range from 0 to num-1.
            label_key (str): The name of the label data in a `.npz` file.
        """
        super().__init__()
        assert isinstance(folder, str)

        self.path = os.path.join(folder, '')

        if isinstance(names, int):
            if names == -1:
                self.names_seq = [f[:-4] for f in os.listdir(folder) if f.endswith('.npz')]
            else:
                self.names_seq = range(names)
        else:
            # NOTE: The number of files must be limited, so the type of `names`
            # provided here must be Sequence instead of Iterable.
            self.names_seq = names

        NUM = len(self.names_seq)
        self._cache = [None, ] * NUM if use_cache else None

    def __len__(self) -> int:
        return len(self.names_seq)

    def has_cache(self, index: int):
        if self._cache is None:
            return False
        return self._cache[index] is not None

    def _read_data(self, fname: Any) -> dict[str, Tensor]:
        file_name = os.path.join(self.path, str(fname) + ".npz")

        with np.load(file_name) as f:
            datadict = dict(f)

        return {key: torch.from_numpy(arr) for key, arr in datadict.items()}

    def _read_batch(self, names: Sequence[Any]):
        return [self._read_data(name) for name in names]

    def __getitem__(self, index) -> dict[str, Tensor]:
        if self.has_cache(index):
            return self._cache[index]
        else:
            pair = self._read_data(self.names_seq[index])

            if self._cache is not None:
                self._cache[index] = pair
            return pair

_KT = TypeVar('_KT')

class MemoryDataset(Dataset):
    def __init__(
            self,
            sample_keys: Sequence[_KT],
            reader: Callable[[_KT], dict[str, Tensor]], *,
            num_workers: int = 0,
            device: Union[str, None] = None,
            pin_memory: Optional[bool] = False,
            tqdm: bool = False
        ) -> None:
        """Preload data from disk to memory, as a Dataset object."""
        super().__init__()
        kwargs = {'device': device, 'pin_memory': pin_memory}
        NUM = len(sample_keys)
        _preloader = DataLoader(dataset=sample_keys,
                                batch_size=None,
                                num_workers=num_workers,
                                collate_fn=reader)
        self._header_data_read = False
        self.data: dict[str, Tensor] = {}

        if tqdm:
            from tqdm import tqdm as _tqdm
            _preloader = _tqdm(_preloader, desc=f"Loading", unit=f'sample')

        for sample_id, pair in enumerate(_preloader):
            if not isinstance(pair, dict):
                raise NotImplementedError("The reader function must return a dict of tensors.")

            if not self._header_data_read: # the first data
                for col in pair: # check the shape of the first data
                    self.data[col] = torch.empty((NUM, *pair[col].shape), dtype=pair[col].dtype, **kwargs)
                    print(f"  {col}: {self.data[col].shape}, {self.data[col].dtype}")
                self._header_data_read = True
 
            for col in pair:
                self.data[col][sample_id].copy_(pair[col], non_blocking=True)

    def __len__(self) -> int:
        return len(self.data[list(self.data.keys())[0]])

    def __getitem__(self, index) -> dict[str, Tensor]:
        return {col: self.data[col][index] for col in self.data.keys()}

    def __getitems__(self, indices) -> dict[str, Tensor]:
        return {col: self.data[col][indices] for col in self.data.keys()}
