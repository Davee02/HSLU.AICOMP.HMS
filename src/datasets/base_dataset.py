import logging
from abc import abstractmethod
from typing import Callable, Optional

import torch

from ..utils.utils import RandomState


class BaseDataset(torch.utils.data.Dataset):
    """
    Initialize a base dataset.

    :param transform: Transformation to apply to the items loaded by the dataset
    :param debug: If debug information should be logged
    :param random_seed: Set class random seed
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        debug: bool = False,
        random_seed: int = 42,
    ):
        super().__init__()
        self.transform = transform
        self.random_seed = random_seed
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def output_signature(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx: int):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def shuffle(self, random_state: RandomState = None):
        """
        Permute (shuffle) the elements in the loader.

        Note that this method will set the `random_state` each time,
        which can produce always the same result
        if called several times with the same argument.

        :param random_state: Random state, int, or generator for shuffling
        :raises NotImplementedError: Not implemented yet
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """
        Return information of the loader.

        Information printed includes a few elements and the total number of samples
        :return: String representation of the loader
        """
        return str(self.__class__.__name__)

    def apply_windowing(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply window function to the signals.

        Note: this method only works BEFORE transforming to frequency domain.

        :param signal: Signal or Signals in the time domain to apply windowing.
        :return: Windowed signal.
        """
        if self.window_fn is not None:
            window = self.window_fn(signal.shape[-1])
            window = window[None, :].repeat(*signal.shape[:-1], 1)
            signal = signal * window
        return signal
