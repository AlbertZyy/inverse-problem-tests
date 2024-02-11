
from typing import (
    Optional, List, Dict, Callable, Iterable, Tuple, Union, Any,
    Generic, TypeVar
)

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

TensorFunc = Callable[[Tensor], Tensor]
LossFunc = Callable[[Tensor, Tensor], Tensor]
ProcessFunc = Callable[[int], Dict[str, float]]
GeneralLoader = Iterable[Tuple[Tensor, Tensor]]

_MT = TypeVar('_MT', bound=Module)
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

def gets(__dict: Dict[_KT, _VT], keys: Iterable[_KT], default: Optional[_VT]=None):
    for key in keys:
        if key in __dict:
            return __dict[key]
    return default

def _read_name(object: Any) -> str:
    if object is None:
        return ''
    return getattr(object, '__name__', None) or getattr(object, '__class__', None).__name__


class ModelManager():
    model: Module

    def __init__(self, script_file: str, args, kwargs, *,
                 checkpoint: Optional[str]=None,
                 map_location: Optional[str]=None,
                 entry_func_name: str='bulid_model',
                 verbose: bool=False) -> None:
        import importlib
        module = importlib.import_module(script_file)
        self.model = module.__dict__[entry_func_name](*args, **kwargs)
        if not isinstance(self.model, Module):
            raise TypeError(f'model is not a torch.nn.Module, but {type(self.model)}')

        self.model_name = _read_name(self.model)
        self.checkpoint = checkpoint

        if verbose:
            print(f'[{self.model_name}]: built from {script_file}')
            print(f'[{self.model_name}]: {self.num_parameters()/1e6:.2f}M parameters')

        if checkpoint is not None:
            try:
                self.load(checkpoint, map_location)
                if verbose:
                    print(f'[{self.model_name}]: checkpoint loaded from {checkpoint}')
            except FileNotFoundError:
                if verbose:
                    print(f'[{self.model_name}]: checkpoint not found: {checkpoint}')

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @classmethod
    def build(cls, config: Dict[str, Any]):
        return cls(**config)

    def load(self, checkpoint: Optional[str]=None, map_location: Optional[str]=None) -> None:
        if checkpoint is None:
            checkpoint = self.checkpoint
        if checkpoint is None:
            raise ValueError('checkpoint is not specified.')

        self.model.load_state_dict(torch.load(checkpoint, map_location))

    def save(self, checkpoint: Optional[str]=None) -> None:
        if checkpoint is None:
            checkpoint = self.checkpoint
        if checkpoint is None:
            raise ValueError('checkpoint is not specified.')

        torch.save(self.model.state_dict(), checkpoint)


class Processor(Generic[_MT]):
    model: _MT
    optim: Optional[Optimizer]
    loss_func: Optional[LossFunc]

    def __init__(self, model: _MT, data_loader: GeneralLoader,
                 optim: Optional[Optimizer]=None,
                 loss_func=None,
                 preprocessor: Optional[TensorFunc]=None, *,
                 tqdm=False,
                 verbose=False) -> None:
        self.model = model
        self.optim = optim
        self.loss_func = loss_func
        self._data_loader = data_loader
        self._preprocessor = preprocessor
        self._tqdm = tqdm
        self._verbose = verbose

    def __repr__(self):
        model_name = _read_name(self.model)
        return f'{self.__class__.__name__}({model_name})'

    def setup(self, config: Dict[str, Any], verbose: Optional[bool]=None) -> None:
        """
        @brief Extract sub-dicts of configures of the optimizer and loss function\
               and setup them.
        """
        if verbose is None:
            verbose = self._verbose

        optim = gets(config, ['optimizer', 'optim', 'Optimizer', 'Optim'], None)
        self.set_optimizer(optim, verbose)
        loss_func = gets(config, ['loss_func', 'loss', 'LossFunc', 'Loss'], None)
        self.set_loss_func(loss_func, verbose)
        return self

    def _read_verbose(self, verbose: Optional[bool]=None):
        if verbose is None:
            verbose = self._verbose
        return verbose

    # NOTE: Construct a optimizer from dict.
    # Example:
    # >>> optim = {'name': 'SGD', 'lr': 0.01, 'momentum': 0.9}
    def set_optimizer(self, optim: Union[Optimizer, Dict[str, Any], None],
                      verbose: Optional[bool]=None) -> None:
        verbose = self._read_verbose(verbose)

        if isinstance(optim, dict):
            name = optim.get('name', 'sgd')
            del optim['name']
            optim_class = getattr(torch.optim, name, None)

            if optim_class is not None:
                self.optim = optim_class(self.model.parameters(), **optim)
                if verbose:
                    print(f'[{self}] Import {name} as the optimizer from torch.optim.')
            else:
                raise NotImplementedError(f'Optimizer type: {name} now is not'
                                          ' found in the torch.optim.')

        else:
            self.optim = optim
            if verbose:
                print(f"[{self}] Optimizer is set to '{_read_name(optim)}'.")

    # NOTE: Construct a loss function from dict.
    # Example:
    # >>> loss_func = {'name': 'CrossEntropyLoss', 'weight': 1.0}
    def set_loss_func(self, loss_func: Union[LossFunc, Module, Dict[str, Any], None],
                      verbose: Optional[bool]=None) -> None:
        verbose = self._read_verbose(verbose)

        if isinstance(loss_func, dict):
            name = loss_func.get('name', None)
            del loss_func['name']
            if name is None:
                raise ValueError('The loss function must have a `name` key.')

            func = getattr(torch.nn.functional, name, None)

            if func is not None:
                self.loss_func = lambda x, y: func(x, y, **loss_func)
                if verbose:
                    print(f'[{self}] Import {name} as the loss function from torch.nn.functional.')
            else:
                func_module = getattr(torch.nn, name, None)

                if func_module is not None:
                    self.loss_func = func_module(**loss_func)
                    if verbose:
                        print(f'[{self}] Import {name} as the loss function from torch.nn.')
                else:
                    raise ValueError(f'Loss function: {name} is not found in torch.nn or'
                                     f' torch.nn.functional.')

        else:
            self.loss_func = loss_func
            if verbose:
                print(f"[{self}] Loss function is set to '{_read_name(loss_func)}'.")

    def set_preprocessor(self, preprocessor: Callable[[Tensor], Tensor],
                         verbose: Optional[bool]=None) -> None:
        verbose = self._read_verbose(verbose)
        self._preprocessor = preprocessor

        if verbose:
            print(f"[{self}] Data preprocessor is set to '{_read_name(preprocessor)}'.")

    # NOTE: This function performs operations for an entire epoch.
    def run(self, epoch: int):
        if self._tqdm:
            from tqdm import tqdm
            iterator = tqdm(self._data_loader, desc=f'Epoch {epoch}', unit='step')
        else:
            iterator = self._data_loader

        for i, (data, label) in enumerate(iterator):
            if self._preprocessor is not None:
                data = self.preprocessor(data)

            result = self.step(i, epoch, data, label)

        if result is None:
            result = {}

        return result

    __call__ = run

    def step(self, step: int, epoch: int, data: Tensor, label: Tensor) -> Dict[str, float]:
        """
        Single step function
        --------------------
        Called in each step. Overide this function.

        1. args are 'step', 'epoch', 'data', 'label'.
        2. 'model', 'optim', and 'loss_func' are reachable in `self`.
        3. 'data' is already preprocessed by the preprocessor.
        4. returns a dict of scalar values to be writen to the tensorboard.

        Don't forget to call `self.optim.step()` and `self.optim.zero_grad()`.
        """
        raise NotImplementedError


class TestingMachine():
    def __init__(self, step_per_epoch: int=1, log_dir: Optional[str]=None, *,
                 max_queue: int=10, flush_secs: int = 120) -> None:
        self._function_list: List[ProcessFunc] = []
        self.step_per_epoch = step_per_epoch
        self.log_dir = log_dir
        self._writer_kwargs = dict(max_queue=max_queue, flush_secs=flush_secs)

    def add_processor(self, processor: ProcessFunc):
        self._function_list.append(processor)

    # NOTE: Do a single epoch. `epoch` is the number of epoch that has been finished
    # when calling this function. Usually it begins from 0.
    def process(self, epoch: int):
        for func in self._function_list:
            result = func(epoch)

            if self._writer is not None:
                for key, value in result.items():
                    # NOTE: the function has been finished, so we use (epoch+1).
                    self._writer.add_scalar(key, value, (epoch+1)*self.step_per_epoch)

    def run(self, n_epoch: int, epoch_head: int, *, tqdm=False):
        if self.log_dir:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(
                log_dir=self.log_dir, **self._writer_kwargs
            )
        else:
            self._writer = None
        epoch_end = epoch_head + n_epoch

        if tqdm:
            from tqdm import trange
            iterator = trange(epoch_head, epoch_end, desc='Processing', unit='epoch')
        else:
            iterator = range(epoch_head, epoch_end)

        for epoch in iterator:
            self.process(epoch)

        if self._writer is not None:
            self._writer.close()
