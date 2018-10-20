"""
example DataLoader that transforms the raw data into the torch DataLoader format
"""
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    def __init__(self, io_paths, hparams, tqdm):
        super().__init__()
        self.tqdm = tqdm
        self.io_paths = io_paths
        self.hparams = hparams
        self.data = {}
        self.loaders = {}
        pass

    @abstractmethod
    def load(self):
        pass
