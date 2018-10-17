"""
Responsible for managing the initialization, training, saving and load of model_saves
"""

import torch
from libs.models import registry
from libs.data_loaders.LanguageLoader import LanguageLoader
from config import basic_hparams
from config import basic_conf as conf
from tqdm import tqdm_notebook
from tqdm import tqdm
import logging

logger = logging.getLogger('__main__')


class ModelManager:
    def __init__(self, hparams=None, path_overrides=None, tqdm_mode='console'):
        logger.info("Initializing Model Manager ...")
        # hyperparameter defaults and path defaults
        self.hparams = basic_hparams.DEFAULT_HPARAMS.copy()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        self.io_paths = conf.DEFAULT_PATHS.copy()
        if isinstance(path_overrides, dict):
            self.io_paths.update(path_overrides)

        # setting the device default
        # todo: need to switch all ref to this, currently still using conf.DEVICE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize other variables we'll definately need
        self.model = None               # for storing the active model that is being trained
        self.model_constructor = None   # the constructor or the active model
        self.loader = None              # currently will be set to LanguageLoader, #todo: think it needs some work

        # todo: implement the following functionality and tracking tools
        self.training_hist = None       # pandas dataframe of acc_history
        self.results = None             # pandas dataframe of saved results
        self.optimizer = None           # memory pointer to optimizer
        self.scheduler = None           # memory pointer to learning rate

        # tqdm mode, set this to notebook or console for the right tqdm
        self.tqdm = tqdm
        self.change_mode(tqdm_mode)

        # other stuff we might need:
        self.data = {}  # memory pointers to the raw data
        self.loaders = {}  # memory pointers to the data loaders

        # logging init
        logger.info("Initialization Complete")
        # logger.info(self.model_details)  # give a lot of attributes with 'None', moved to self.reset()

    @property
    def model_details(self):
        info = '\n*********** Training Model Details ***********'
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], dict):
                for sub_key in self.__dict__[key].keys():
                    info += "\n-- self.%s.%s = %s" % (key, sub_key, str(self.__dict__[key][sub_key]))
            else:
                info += "\n-- self.%s = %s" % (key, str(self.__dict__[key]))
        info += '\n************ End of Model Details ************'
        return info

    def load_data(self):
        logger.info("loading data into LanguageLoader")
        self.loader = LanguageLoader(self.io_paths['input_lang'],
                                     self.io_paths['output_lang'],
                                     self.hparams['vocab_size'],
                                     self.hparams['max_length'])

    def save_model(self):
        """ saves the active model to the io_path specification """
        self.model.save(self.io_paths)

    def load_model(self):
        """ loads a model from the io_path specification """
        # todo: implement loading

    def set_model(self, model_name):
        """ finds the constructor in libs.models and initializes the model """
        self.model_constructor = registry.reg[model_name]
        self.reset()

    def change_mode(self, tqdm_mode):
        """ changes between notebook or console mode """
        if tqdm_mode == 'console':
            self.tqdm = tqdm
        else:
            self.tqdm = tqdm_notebook

    def reset(self):
        """ reinitializes the active model by calling the constructor """
        # todo: the model should have no dependency on the loader, these dependancies need to be removed
        self.model = self.model_constructor(self.loader.input_size, self.loader.output_size)
        # report model details after each reset
        logger.info(self.model_details)

    def train(self):
        """ continue to train the current model """
        losses = []
        for epoch in self.tqdm(range(self.hparams["num_epochs"])):
            logger.info("{} epoch: {} {}".format("=" * 20, epoch, "=" * 20))
            for i, batch in enumerate(self.loader.sentences(self.hparams["num_batches"])):
                input_, target = batch

                loss, outputs = self.model.train(input_, target)
                losses.append(loss)

                if i % 100 == 0:
                    msg = "Loss at step {}: {:.2f}".format(i, loss)
                    msg += "\nTruth: \"{}\"".format(self.loader.vec_to_sentence(target))
                    msg += "\nGuess: \"{}\"\n".format(
                        self.loader.vec_to_sentence(outputs[:-1]))
                    logger.info(msg)
                    self.save_model()


# todo: to be used later to initialize the pandas dataframe results output
def _init_cur_res():
    """ initializes all results to '',
    does not need HPARAMS already defined in the config, those are automatically added later """
    return {
        'initial_val_acc': '',
        'final_val_acc': '',
        'training_time': '',
        'early_stopped': ''
    }


# todo: I'm not sure what this method is doing, saving it here for now, aim is to deprecate
def translate(data, rnn):
    """
    translate()
    """
    vecs = data.sentence_to_vec("the president is here <EOS>")

    translation = rnn.eval(vecs)
    print(data.vec_to_sentence(translation))