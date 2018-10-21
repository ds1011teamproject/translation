"""
Responsible for managing the initialization, training, saving and load of model_saves
"""
import os
import pandas as pd
import gc
from libs.models import registry as mod_reg
from libs.data_loaders import registry as load_reg
from libs.models.registry import modelRegister
from libs.data_loaders.registry import loaderRegister
import matplotlib.pyplot as plt
from config.constants import PathKey, LoadingKey
from config import basic_hparams
from config import basic_conf as conf
from tqdm import tqdm_notebook
from tqdm import tqdm
from string import punctuation
import logging
from libs._version import __version__

logger = logging.getLogger('__main__')


class ModelManager:
    """
    the model manager can operate in either console mode or notebook mode

    ## console mode:
    - Uses the console version of tqdm to display progress on various dataloading or training iterations
    - Saves any output to the model_saves folder under the subfolder of the model's name

    ## notebook mode:
    - uses the notebook version of tqdm that integrates with jupyter
    - shows output to the notebook when it can
    """
    OPT_MODE_CONSOLE = 'console'
    OPT_MODE_NOTEBOOK = 'notebook'
    GRAPH_MODE_TRAINING = 'training'
    GRAPH_MODE_EPOCH = 'epoch'

    def __init__(self, hparams=None, control_overrides=None, mode=OPT_MODE_CONSOLE):
        logger.info("Initializing Model Manager, version %s ..." % __version__)
        # hyperparameter defaults and path defaults
        self.hparams = _update_if_dict(basic_hparams.DEFAULT_HPARAMS, hparams)
        self.cparams = _update_if_dict(conf.DEFAULT_CONTROLS, control_overrides)
        self.lparams = None             # the loader will send back a list of parameter required for model init
        self.model = None               # for storing the active model that is being trained
        self.dataloader = None          # currently will be set a default DataLoader used for HW1
        self.results = []               # list of saved results, can be output to a pandas dataframe w/ get_results
        self.mode = mode                # able to operate in console or notebook mode
        self.tqdm = tqdm                # memory ref to the tqdm handler in the correct mode
        self.change_mode(mode)          # setting the right mode fo the tqdm handler

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
        """
        The load_data method should be called right after manager init, and before model init.
        This is because the model initializtion requires information from the load routine
        :param loader_name: name that should appear in the loader registry libs.data_loaders.registry.reg
        """
        logger.info("Loading data using %s ..." % loader_name)
        self.dataloader = load_reg.reg[loader_name](self.cparams, self.hparams, self.tqdm)
        # the load routine should return a dict of parameters that models need to init
        self.lparams = self.dataloader.load()

    def new_model(self, model_name, label='scratch', nolog=False):
        """
        initializes a new model, the model_ref should appear in the model registry
        :param model_name: should appear in the model registry libs.models.registry.reg
        :param label: label associated with the model, a folder in model_saves will be created for this label
        :param nolog: suppresses the logger in the model constructor
        """
        safe_label = _enforce_folder_label(label)
        if self.dataloader is None:
            logger.warning("Starting a new model with no data loaded, "
                           "models usually require information about the data to initialize!")

        cur_constructor = mod_reg.reg[model_name]
        model_path = os.path.join(self.cparams[PathKey.MODEL_SAVES], safe_label + os.sep)
        self.cparams[PathKey.MODEL_PATH] = model_path
        self.model = cur_constructor(self.hparams, self.lparams, self.cparams, safe_label, nolog=nolog)

        # make the directory for model saves:
        os.makedirs(model_path, exist_ok=True)
        logger.info("New Model initialized: /%s, all model output files will be saved here: %s" % (label, model_path))

    def save_model(self, md_string, fn=None):
        """ explicitly saves the active model to the cparams specification """
        self.model.save(fn)
        self.model.save_meta(md_string)

    def load_model(self, which_model=LoadingKey.LOAD_CHECKPOINT, path_to_model_ovrd=None):
        """
        loads a model from the cparams specification
        :param which_model: 'checkpoint' or 'best'
        :param path_to_model_ovrd: override path to file
        """
        self.model.load(which_model=which_model, path_to_model_ovrd=path_to_model_ovrd)

    def train(self):
        """ continue to train the current model """
        self.model.train(self.dataloader, self.tqdm)
        self.results.append(self.model.output_dict)

    def dump_model(self):
        """ dumps the existing model to clear up memory """
        self.model = None
        gc.collect()

    def graph_training_curves(self, mode=GRAPH_MODE_TRAINING):
        """
        graphs the training curves that are saved in the active model
        if the manager is operating in console mode, the files are saved in a png in the model dir
        :param mode: in training mode or epoch mode, determines frequency of the x axis (by check iteration or by epoch)
        """
        if self.model is None:
            logger.error("cannot graph training curves since there is no trained model")
        else:
            if mode == self.GRAPH_MODE_TRAINING:
                curves = self.model.iter_curves
                xlab = 'model check iterations'
            else:
                curves = self.model.epoch_curves
                xlab = 'model epoch'
            f, ax = plt.subplots(1, 2, figsize=(15, 7))
            ax[0].plot(curves[self.model.TRAIN_ACC], label='training acc')
            ax[0].plot(curves[self.model.VAL_ACC], label='val acc')
            ax[0].set_xlabel(xlab)
            ax[0].legend()

            ax[1].plot(curves[self.model.TRAIN_LOSS], label='training loss')
            ax[1].plot(curves[self.model.VAL_LOSS], label='val loss')
            ax[1].legend()
            ax[1].set_xlabel(xlab)

            if self.mode == self.OPT_MODE_CONSOLE:
                fn = self.cparams[PathKey.MODEL_PATH] + mode + '_curves.png'
                plt.savefig(fn)
                logger.info("Model %s curve chart saved to %s" % (mode, fn))
            else:
                plt.show()

    def change_mode(self, tqdm_mode):
        """ changes between notebook or console mode """
        if tqdm_mode == self.OPT_MODE_CONSOLE:
            self.tqdm = tqdm
        else:
            self.tqdm = tqdm_notebook

    def get_results(self):
        """ outputs the self.results list of collected training results into a dataframe """
        return pd.DataFrame(self.results)


def _update_if_dict(default_dict, overrides):
    """ used by the init routine to update the hparam and cparams dicts """
    rt_dict = default_dict.copy()
    if isinstance(overrides, dict):
        rt_dict.update(overrides)
    return rt_dict


def _enforce_folder_label(label):
    """ enforces small caps and no punc in the label so we can create a folder with the string """
    label = label.replace(" ", "_")
    translator = label.maketrans('', '', punctuation)
    label = label.translate(translator)
    return label
