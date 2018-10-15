"""
Responsible for managing the initialization, training, saving and load of models
controls:
- load pickles bool
- save pickles bool
- maintain best model save
- force device
- encoder Constructor
- decoder Constructor
- hyperparameters
- save results bool
- result outpath
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
