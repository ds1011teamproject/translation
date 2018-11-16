"""
super class for all of the various model we will build

Notes:
- the model should implement its own training loop / early stop
- it should track its own training curve
- it should track its own results

"""
from abc import ABC, abstractmethod
from config.constants import HyperParamKey, ControlKey, StateKey, PathKey, LoadingKey, OutputKey
import logging
import torch
import time

logger = logging.getLogger('__main__')


class BaseModel(ABC):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    TRAIN_ACC = 'train_acc'
    TRAIN_LOSS = 'train_loss'
    VAL_ACC = 'val_acc'
    VAL_LOSS = 'val_loss'
    CHECKPOINT_FN = 'checkpoint.tar'
    BEST_FN = 'model_best.tar'
    README_FN = 'README.md'

    def __init__(self, hparams, lparams, cparams, label='scratch', nolog=False):
        super().__init__()
        self.label = label
        self.hparams = hparams      # model hyperparameters
        # self.lparams = lparams      # model loader parameters
        self.cparams = cparams      # model control parameters
        self.cur_epoch = 0          # tracks the current epoch (for saving/loading and scheduler) 0 indexed

        self.iter_curves = dict()
        self.epoch_curves = dict()
        self.output_dict = {}       # key value pair of any results you want to save

        self._write_params_to_output_dict()
        if not nolog:
            logger.info(self.model_details)

    @property
    def model_details(self):
        info = '\n*********** Model: %s Details ***********' % self.label
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], dict):
                for sub_key in self.__dict__[key].keys():
                    info += "\n-- self.%s.%s = %s" % (key, sub_key, str(self.__dict__[key][sub_key]))
            else:
                info += "\n-- self.%s = %s" % (key, str(self.__dict__[key]))
        info += '\n************ End of Model: %s Details ************' % self.label
        return info

    def save_meta(self, meta_string):
        """ saves self.meta to a md file, appends all parameters"""
        fp = self.cparams[PathKey.MODEL_PATH] + self.README_FN
        f = open(fp, 'w')
        f.write(meta_string)
        f.write('\n\nHyperparameters used:')
        for key in self.hparams.keys():
            f.write("\n%s - %s" % (key, self.hparams[key]))
        f.write('\n\nLoader parameters used:')
        # for key in self.lparams.keys():
        #     f.write("\n%s - %s" % (key, self.lparams[key]))
        f.write('\n\nControl parameters used:')
        for key in self.cparams.keys():
            f.write("\n%s - %s" % (key, self.cparams[key]))
        f.close()

    def add_epochs(self, num_added):
        self.hparams[HyperParamKey.NUM_EPOCH] += num_added
        logger.info("added %s to required epochs count. \n"
                    "cur epoch=%s, required epochs=%s" % (num_added
                                                          , self.cur_epoch
                                                          , self.hparams[HyperParamKey.NUM_EPOCH]))

    def _save_checkpoint(self, state, filename):
        start_time = time.time()
        save_path = self.cparams[PathKey.MODEL_PATH]
        logger.debug("Saving Model to %s" % (save_path + filename))
        torch.save(state, save_path + filename)
        logger.debug("Saving took %.2f secs" % (time.time() - start_time))

    def _write_params_to_output_dict(self):
        """ writes hparams, lparams to the output_dict """
        if isinstance(self.hparams, dict):
            self.output_dict.update(self.hparams)
        # if isinstance(self.lparams, dict):
        #     self.output_dict.update(self.lparams)

    def _init_optim_and_scheduler(self):
        self._init_optim()
        self._init_scheduler()

    @abstractmethod
    def _init_optim(self):
        pass

    @abstractmethod
    def _init_scheduler(self):
        pass

    @abstractmethod
    def train(self, dataloader, tqdm_handler):
        """
        Train loop varies with different types of problems.
        e.g. classification, inference, machine translation, etc.
        """
        pass

    @abstractmethod
    def eval_model(self, dataloader):
        """
        Evaluate the model at some point of training progress.
        Reporting metrics such as train/validation losses, accuracies, etc.
        :param dataloader: pytorch DataLoader, created from the data set to be evaluated
        """
        pass

    @abstractmethod
    def check_early_stop(self):
        """
        Set the flag of early-stopping in training process.
        :return: True/False
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save the model, optimizer(s), lr_scheduler(s) to checkpoint file.
        State dicts varies w.r.t. different types of problems.
        """
        pass

    @abstractmethod
    def load(self):
        """
        Load the model, optimizer, lr_scheduler from checkpoints (inverse of save method)
        The state dicts varies w.r.t. problems and models.
        """
        pass
