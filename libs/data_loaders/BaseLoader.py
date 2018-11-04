"""
Base Class for all loader handlers

- stores raw data in self.data = {}, with keys train, val, test
- stores transformed DataLoader objects in self.loaders = {}, with keys train, val, test
"""
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    def __init__(self, cparams, hparams, tqdm):
        super().__init__()
        self.tqdm = tqdm            # tqdm handler that depends on console or notebook mode
        self.cparams = cparams      # cparams passed down from the ModelManager
        self.hparams = hparams      # hparams passed down from the ModelManager
        self.data = {}              # for storing raw data
        self.loaders = {}           # for storing torch DataLoader objects

    @abstractmethod
    def load(self):
        """
        this method on the child class should load the data into self.data and self.loaders, also returns lparams
        :return: lparams is a dict with parameters that the model constructure will need later (likely input_size)
        """
        pass
