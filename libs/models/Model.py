"""
super class for all of the various model we will build

Notes:
- the model should implement its own training loop / early stop
- it should track its own training curve
- it should track its own results

"""
from abc import ABC, abstractmethod
from config.constants import HyperParamKey
import logging

logger = logging.getLogger('__main__')


class Model(ABC):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    TRAIN_ACC = 'train_acc'
    TRAIN_LOSS = 'train_loss'
    VAL_ACC = 'val_acc'
    VAL_LOSS = 'val_loss'

    def __init__(self, hparams, lparams, io_paths, alias='scratch'):
        super().__init__()
        self.name = alias
        self.hparams = hparams
        self.lparams = lparams
        self.io_paths = io_paths
        self.model = None  # used to store the NN graph

        # used for storing the training acc/loss curves by each model evaluation
        self.train_curves = {
            self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            self.VAL_ACC: [],
            self.VAL_LOSS: [],
        }

        # used for storing the training acc/loss curves by each model epoch
        self.epoch_curves = {
            self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            self.VAL_ACC: [],
            self.VAL_LOSS: [],
        }

        self.hparams = hparams          # passed down from the ModelManager
        self.output_dict = {}           # key value pair of any results you want to save, collected by Modelmanager

        logger.info(self.model_details)

    @property
    def model_details(self):
        info = '\n*********** Model: %s Details ***********' % self.name
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], dict):
                for sub_key in self.__dict__[key].keys():
                    info += "\n-- self.%s.%s = %s" % (key, sub_key, str(self.__dict__[key][sub_key]))
            else:
                info += "\n-- self.%s = %s" % (key, str(self.__dict__[key]))
        info += '\n************ End of Model: %s Details ************' % self.name
        return info

    def train(self, loader, tqdm_handler):
        """
        this is a basic training loop, that can be overloaded if you need to tweak it
        :param loader: torch.utils.DataLoader, filled with data in the load routine
        :param tqdm_handler:
        :return:
        """
        if self.model is None:
            logger.error("Cannot train an uninitialized model - stopping training on model %s" % self.name)
        else:
            # ------ basic training loop ------
            op_constr = self.hparams[HyperParamKey.OPTIMIZER_ENCODER]
            optimizer = op_constr(self.model.parameters())  # todo: add optimizer hparameters

            sche_constr = self.hparams[HyperParamKey.SCHEDULER]
            scheduler = sche_constr(optimizer, gamma=self.hparams[HyperParamKey.SCHEDULER_GAMMA])

            criterion = self.hparams[HyperParamKey.CRITERION]()

            early_stop_training = False
            for epoch in range(self.hparams[HyperParamKey.NUM_EPOCH]):
                scheduler.step()
                for i, (data_batch, length_batch, label_batch) in enumerate(loader.loaders['train']):
                    self.model.train()  # good practice to set the model to training mode (enables dropout)
                    optimizer.zero_grad()
                    outputs = self.model(data_batch, length_batch)  # forward pass
                    loss = criterion(outputs, label_batch)          # computing loss
                    loss.backward()                                 # backprop
                    optimizer.step()                                # taking a step

                    # --- Model Evaluation ---
                    if (i + 1) % self.hparams[HyperParamKey.TRAIN_LOOP_EVAL_FREQ] == 0:
                        val_acc, val_loss = self.eval_model(loader.loaders['val'])
                        train_acc, train_loss = self.eval_model(loader.loaders['train'])

                        logger.info('Ep:%s, Bt:%s/%s, VAcc:%.2f, VLoss:%.1f, TAcc:%.2f, TLoss:%.1f, LR:%.4f' %
                                    (epoch + 1,
                                     i + 1,
                                     len(loader.loaders['train']),
                                     val_acc,
                                     val_loss,
                                     train_acc,
                                     train_loss,
                                     optimizer.param_groups[0]['lr'])  # assumes a constant lr across params
                                    )
                        self.train_curves[self.TRAIN_LOSS].append(train_loss)
                        self.train_curves[self.TRAIN_ACC].append(train_acc)
                        self.train_curves[self.VAL_LOSS].append(val_loss)
                        self.train_curves[self.VAL_ACC].append(val_acc)

                        if self.hparams[HyperParamKey.CHECK_EARLY_STOP]:
                            early_stop_training = self.check_early_stop()

                        if early_stop_training:
                            logger.info('--- stopping training due to early stop ---')
                            break

                    if early_stop_training:
                        break

                # appending to epock trackers
                val_acc, val_loss = self.eval_model(loader.loaders['val'])
                train_acc, train_loss = self.eval_model(loader.loaders['train'])
                self.epoch_curves[self.TRAIN_LOSS].append(train_loss)
                self.epoch_curves[self.TRAIN_ACC].append(train_acc)
                self.epoch_curves[self.VAL_LOSS].append(val_loss)
                self.epoch_curves[self.VAL_ACC].append(val_acc)

    def plot_curves(self):
        """ draws the training and validation curves """

    @abstractmethod
    def eval_model(self, dataloader):
        # needs to return a tuple of (acc, loss)
        pass

    @abstractmethod
    def check_early_stop(self):
        # returns whether we can early stop the model
        pass

    @abstractmethod
    def save(self, io_paths):
        pass
