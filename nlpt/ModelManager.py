"""
Responsible for managing the initialization, training, saving and load of models
"""


class ModelManager:
    def __init__(self):
        self.model = None

    def save(self):
        """ saves the active model """

    def load(self, path):
        """ loads a model from path"""

    def train(self, data_loader):
        """ continue to train the current model """
