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
        self.lparams = lparams      # model loader parameters
        self.cparams = cparams      # model control parameters
        self.cur_epoch = 0          # tracks the current epoch (for saving/loading and scheduler) 0 indexed
        self.model = None           # mem pointer to the nn.Module
        self.optim = None           # mem pointer to the optim
        self.scheduler = None       # mem pointer to the lr_scheduler

        # used for storing the training acc/loss curves by each model evaluation
        self.iter_curves = {
            self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            self.VAL_ACC: [],
            self.VAL_LOSS: [],
        }

        # used for storing the curves by each model epoch
        self.epoch_curves = {
            self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            self.VAL_ACC: [],
            self.VAL_LOSS: [],
        }
        self.hparams = hparams                          # passed down from the ModelManager
        self.output_dict = {}                           # key value pair of any results you want to save
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

    def train(self, loader, tqdm_handler):
        """
        this is a basic training loop, that can be overloaded if you need to tweak it
        :param loader: torch.utils.DataLoader, filled with data in the load routine
        :param tqdm_handler:
        :return:
        """
        if self.model is None:
            logger.error("Cannot train an uninitialized model - stopping training on model %s" % self.label)
        else:
            # ------ basic training loop ------
            self._init_optim_and_scheduler()
            criterion = self.hparams[HyperParamKey.CRITERION]()

            early_stop_training = False
            for epoch in tqdm_handler(range(self.hparams[HyperParamKey.NUM_EPOCH] - self.cur_epoch)):
                self.scheduler.step(epoch=self.cur_epoch)  # scheduler calculates the lr based on the cur_epoch
                self.cur_epoch += 1
                logger.info("stepped scheduler to epoch = %s" % str(self.scheduler.last_epoch + 1))

                for i, (data_batch, length_batch, label_batch) in enumerate(loader.loaders['train']):
                    self.model.train()  # good practice to set the model to training mode (enables dropout)
                    self.optim.zero_grad()
                    outputs = self.model(data_batch, length_batch)  # forward pass
                    loss = criterion(outputs, label_batch)          # computing loss
                    loss.backward()                                 # backprop
                    self.optim.step()                               # taking a step

                    # --- Model Evaluation Iteration ---
                    is_best = False
                    if (i + 1) % self.hparams[HyperParamKey.TRAIN_LOOP_EVAL_FREQ] == 0:
                        val_acc, val_loss = self.eval_model(loader.loaders['val'])
                        train_acc, train_loss = self.eval_model(loader.loaders['train'])
                        iter_curve = self.iter_curves[self.VAL_ACC]
                        if len(iter_curve) > 0 and val_acc >= max(iter_curve):
                            is_best = True

                        logger.info('Ep:%s/%s, Bt:%s/%s, VAcc:%.2f, VLoss:%.1f, TAcc:%.2f, TLoss:%.1f, LR:%.4f' %
                                    (self.cur_epoch,
                                     self.hparams[HyperParamKey.NUM_EPOCH],
                                     i + 1,
                                     len(loader.loaders['train']),
                                     val_acc,
                                     val_loss,
                                     train_acc,
                                     train_loss,
                                     self.optim.param_groups[0]['lr'])  # assumes a constant lr across params
                                    )
                        self.iter_curves[self.TRAIN_LOSS].append(train_loss)
                        self.iter_curves[self.TRAIN_ACC].append(train_acc)
                        self.iter_curves[self.VAL_LOSS].append(val_loss)
                        self.iter_curves[self.VAL_ACC].append(val_acc)
                        if self.cparams[ControlKey.SAVE_BEST_MODEL] and is_best:
                            self.save(fn=self.BEST_FN)

                        # reporting back up to output_dict

                        if is_best:
                            self.output_dict[OutputKey.BEST_VAL_ACC] = val_acc
                            self.output_dict[OutputKey.BEST_VAL_LOSS] = val_loss

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
                if self.cparams[ControlKey.SAVE_EACH_EPOCH]:
                    self.save()

            # final loss reporting
            val_acc, val_loss = self.eval_model(loader.loaders['val'])
            train_acc, train_loss = self.eval_model(loader.loaders['train'])
            self.output_dict[OutputKey.FINAL_TRAIN_ACC] = train_acc
            self.output_dict[OutputKey.FINAL_TRAIN_LOSS] = train_loss
            self.output_dict[OutputKey.FINAL_VAL_ACC] = val_acc
            self.output_dict[OutputKey.FINAL_VAL_LOSS] = val_loss
            logger.info("training completed, results collected ...")

    def save_meta(self, meta_string):
        """ saves self.meta to a md file, appends all parameters"""
        fp = self.cparams[PathKey.MODEL_PATH] + self.README_FN
        f = open(fp, 'w')
        f.write(meta_string)
        f.write('\n\nHyperparameters used:')
        for key in self.hparams.keys():
            f.write("\n%s - %s" % (key, self.hparams[key]))
        f.write('\n\nLoader parameters used:')
        for key in self.lparams.keys():
            f.write("\n%s - %s" % (key, self.lparams[key]))
        f.write('\n\nControl parameters used:')
        for key in self.cparams.keys():
            f.write("\n%s - %s" % (key, self.cparams[key]))
        f.close()

    def save(self, fn=None):
        """
        saves the full structure for future loading
        """
        state = {
            StateKey.MODEL_STATE:   self.model.state_dict(),
            StateKey.OPTIM_STATE:   self.optim.state_dict(),
            StateKey.SCHED_STATE:   self.scheduler.state_dict(),
            StateKey.HPARAMS:       self.hparams,
            StateKey.LPARAMS:       self.lparams,
            StateKey.CPARAMS:       self.cparams,
            StateKey.ITER_CURVES:   self.iter_curves,
            StateKey.EPOCH_CURVES:  self.epoch_curves,
            StateKey.CUR_EPOCH:     self.cur_epoch,
            StateKey.LABEL:         self.label
        }

        if fn is None:
            self._save_checkpoint(state)
        else:
            self._save_checkpoint(state, fn)

    def load(self, which_model=LoadingKey.LOAD_CHECKPOINT, path_to_model_ovrd=None):
        """
        can load either the best model, the checkpoint or a specific path
        :param which_model: 'checkpoint' or 'best'
        :param path_to_model_ovrd: override path to file
        """
        if path_to_model_ovrd is None:
            if which_model == LoadingKey.LOAD_BEST:
                load_path = self.cparams[PathKey.MODEL_PATH] + self.BEST_FN
            else:
                # loads the check point by default
                load_path = self.cparams[PathKey.MODEL_PATH] + self.CHECKPOINT_FN
        else:
            load_path = path_to_model_ovrd

        logger.info("loading checkpoint at %s" % load_path)
        loaded = torch.load(load_path)

        self.model.load_state_dict(loaded[StateKey.MODEL_STATE])
        self._init_optim()
        self.optim.load_state_dict(loaded[StateKey.OPTIM_STATE])
        self._init_scheduler()
        self.scheduler.load_state_dict(loaded[StateKey.SCHED_STATE])
        self.hparams = loaded[StateKey.HPARAMS]
        self.lparams = loaded[StateKey.LPARAMS]
        self.cparams = loaded[StateKey.CPARAMS]
        self.iter_curves = loaded[StateKey.ITER_CURVES]
        self.epoch_curves = loaded[StateKey.EPOCH_CURVES]
        self.cur_epoch = loaded[StateKey.CUR_EPOCH]
        self.label = loaded[StateKey.LABEL]

        logger.info("Successfully loaded checkpoint!")

    def add_epochs(self, num_added):
        self.hparams[HyperParamKey.NUM_EPOCH] += num_added
        logger.info("added %s to required epochs count. \n"
                    "cur epoch=%s, required epochs=%s" % (num_added
                                                          , self.cur_epoch
                                                          , self.hparams[HyperParamKey.NUM_EPOCH]))

    def _init_optim_and_scheduler(self):
        self._init_optim()
        self._init_scheduler()

    def _init_optim(self):
        op_constr = self.hparams[HyperParamKey.OPTIMIZER_ENCODER]
        self.optim = op_constr(self.model.parameters(), lr=self.hparams[HyperParamKey.LR])

    def _init_scheduler(self):
        sche_constr = self.hparams[HyperParamKey.SCHEDULER]
        self.scheduler = sche_constr(self.optim, gamma=self.hparams[HyperParamKey.SCHEDULER_GAMMA])

    def _save_checkpoint(self, state, filename=CHECKPOINT_FN):
        start_time = time.time()
        save_path = self.cparams[PathKey.MODEL_PATH]
        logger.debug("Saving Model to %s" % (save_path + filename))
        torch.save(state, save_path + filename)
        logger.debug("Saving took %.2f secs" % (time.time() - start_time))

    def _write_params_to_output_dict(self):
        """ writes hparams, lparams to the output_dict """
        if isinstance(self.hparams, dict):
            self.output_dict.update(self.hparams)
        if isinstance(self.lparams, dict):
            self.output_dict.update(self.lparams)

    @abstractmethod
    def eval_model(self, dataloader):
        # needs to return a tuple of (acc, loss)
        pass

    @abstractmethod
    def check_early_stop(self):
        # returns whether we can early stop the model
        pass
