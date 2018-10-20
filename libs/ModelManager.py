"""
Responsible for managing the initialization, training, saving and load of model_saves
"""
import os
from libs.models import registry as mod_reg
from libs.data_loaders import registry as load_reg
from libs.models.registry import modelRegister
from libs.data_loaders.registry import loaderRegister
import matplotlib.pyplot as plt
from config.constants import HyperParamKey, PathKey
from config import basic_hparams
from config import basic_conf as conf
from tqdm import tqdm_notebook
from tqdm import tqdm
import logging

logger = logging.getLogger('__main__')


class ModelManager:
    def __init__(self, hparams=None, path_overrides=None, mode='console'):
        logger.info("Initializing Model Manager ...")
        # hyperparameter defaults and path defaults
        self.hparams = _update_if_dict(basic_hparams.DEFAULT_HPARAMS, hparams)
        self.io_paths = _update_if_dict(conf.DEFAULT_PATHS, path_overrides)
        self.lparams = None             # the loader will send back a list of parameter required for model init

        # initialize other variables we'll definitely need
        self.model = None               # for storing the active model that is being trained
        self.model_path = None          # for storing any files that the model might save
        self.loader = None              # currently will be set a default DataLoader used for HW1
        self.results = None             # pandas dataframe of saved results

        # tqdm mode, set this to notebook or console for the right tqdm
        self.mode = mode
        self.tqdm = tqdm
        self.change_mode(mode)

        # other stuff we might need:
        self.data = {}                  # memory pointers to the raw data
        self.loaders = {}               # memory pointers to the data loaders
        logger.info(modelRegister.model_list)
        logger.info(loaderRegister.loader_list)
        logger.info(self.manager_details)

    @property
    def manager_details(self):
        info = '\n*********** Model Manager Details ***********'
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], dict):
                for sub_key in self.__dict__[key].keys():
                    info += "\n-- self.%s.%s = %s" % (key, sub_key, str(self.__dict__[key][sub_key]))
            else:
                info += "\n-- self.%s = %s" % (key, str(self.__dict__[key]))
        info += '\n************ End of Model Manager Details ************'
        return info

    def load_data(self, loader_name):
        logger.info("Loading data using the %s ..." % loader_name)
        self.loader = load_reg.reg[loader_name](self.io_paths, self.hparams, self.tqdm)
        self.lparams = self.loader.load()

    def save_model(self):
        """ saves the active model to the io_path specification """
        self.model.save(self.io_paths)

    def load_model(self):
        """ loads a model from the io_path specification """

    def new_model(self, model_ref, label='scratch'):
        """ finds the constructor in libs.models and initializes the model """
        if self.loader is None:
            logger.warning("Starting a new model with no data loaded, "
                           "models usually require information about the data to initialize!")

        cur_constructor = mod_reg.reg[model_ref]
        self.model = cur_constructor(self.hparams, self.lparams, self.io_paths, label)

        # make the directory for model saves:
        self.model_path = os.path.join(self.io_paths[PathKey.MODEL_SAVES], label + os.sep)
        os.makedirs(self.model_path, exist_ok=True)
        logger.info("All model output files will be saved here: %s" % self.model_path)

    def train(self):
        """ continue to train the current model """
        self.model.train(self.loader, self.tqdm)

    def graph_training_curves(self):
        if self.model is None:
            logger.error("cannot graph training curves since there is no trained model")
        else:
            f, ax = plt.subplots(1, 2, figsize=(15, 7))
            ax[0].plot(self.model.train_curves['train_acc'], label='training acc')
            ax[0].plot(self.model.train_curves['val_acc'], label='val acc')
            ax[0].set_xlabel('model check iterations')
            ax[0].legend()

            ax[1].plot(self.model.train_curves['train_loss'], label='training loss')
            ax[1].plot(self.model.train_curves['val_loss'], label='val loss')
            ax[1].legend()
            ax[1].set_xlabel('model check iterations')

            if self.mode == 'console':
                fn = self.model_path + 'training_curves.png'
                plt.savefig(fn)
                logger.info("Model training curve chart saved to %s" % fn)
            else:
                plt.show()

    def change_mode(self, tqdm_mode):
        """ changes between notebook or console mode """
        if tqdm_mode == 'console':
            self.tqdm = tqdm
        else:
            self.tqdm = tqdm_notebook


def _update_if_dict(default_hparams, overrides):
    rt_dict = default_hparams.copy()
    if isinstance(overrides, dict):
        rt_dict.update(overrides)
    return rt_dict
