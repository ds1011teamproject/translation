"""
super class for all of the various model we will build
"""
from abc import ABC, abstractmethod


class TranslationModel(ABC):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def train(self, input_, target):
        pass

    @abstractmethod
    def eval(self, input_):
        pass

    @abstractmethod
    def save(self, io_paths):
        pass