"""
Base model for document classification.
Detailed models (e.g. BagOfWord) inherit the CBaseModel, sharing
methods like `train`, `save`, and 'load`, while having different
evaluation and early-stop methods.

"""
import logging
import torch
import torch.nn.functional as F

from libs.models.BaseModel import BaseModel
from config.constants import (HyperParamKey, ControlKey, PathKey,
                              StateKey, LoadingKey, OutputKey)

logger = logging.getLogger('__main__')


class CBaseModel(BaseModel):

    def __init__(self, hparams, lparams, cparams, label='scratch', nolog=False):
        super().__init__(hparams, lparams, cparams, label, nolog)
        self.model = None  # mem pointer to the nn.Module
        self.optim = None  # mem pointer to the optim
        self.scheduler = None  # mem pointer to the lr_scheduler

        # storing the training acc/loss curves by each model evaluation:
        self.iter_curves = {
            self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            self.VAL_ACC: [],
            self.VAL_LOSS: [],
        }
        # storing the curves by each model epoch:
        self.epoch_curves = {
            self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            self.VAL_ACC: [],
            self.VAL_LOSS: [],
        }

    def train(self, loader):
        """
        this is a basic training loop, that can be overloaded if you need to tweak it
        :param loader: torch.utils.DataLoader, filled with data in the load routine
        :return:
        """
        if self.model is None:
            logger.error("Cannot train an uninitialized model - stopping training on model %s" % self.label)
        else:
            # ------ basic training loop ------
            self._init_optim_and_scheduler()
            criterion = self.hparams[HyperParamKey.CRITERION]()

            early_stop_training = False
            for epoch in range(self.hparams[HyperParamKey.NUM_EPOCH] - self.cur_epoch):
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
                if early_stop_training:
                        break

            # final loss reporting
            val_acc, val_loss = self.eval_model(loader.loaders['val'])
            train_acc, train_loss = self.eval_model(loader.loaders['train'])
            self.output_dict[OutputKey.FINAL_TRAIN_ACC] = train_acc
            self.output_dict[OutputKey.FINAL_TRAIN_LOSS] = train_loss
            self.output_dict[OutputKey.FINAL_VAL_ACC] = val_acc
            self.output_dict[OutputKey.FINAL_VAL_LOSS] = val_loss
            logger.info("training completed, results collected ...")

    def save(self, fn=BaseModel.CHECKPOINT_FN):
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

    def _init_optim(self):
        op_constr = self.hparams[HyperParamKey.OPTIMIZER]
        self.optim = op_constr(self.model.parameters(), lr=self.hparams[HyperParamKey.ENC_LR])

    def _init_scheduler(self):
        sche_constr = self.hparams[HyperParamKey.SCHEDULER]
        self.scheduler = sche_constr(self.optim, gamma=self.hparams[HyperParamKey.SCHEDULER_GAMMA])

    ################################
    # Override following if needed #
    ################################

    def eval_model(self, dataloader):
        """
        takes all of the data in the loader and forward passes through the model
        :param dataloader: the torch.utils.data.DataLoader with the data to be evaluated
        :return: tuple of (accuracy, loss)
        """
        if self.model is None:
            raise AssertionError("cannot evaluate model: %s, it was never initialized" % self.label)
        else:
            correct = 0
            total = 0
            cur_loss = 0
            self.model.eval()  # good practice to set the model to evaluation mode (no dropout)
            for data, lengths, labels in dataloader:
                data_batch, length_batch, label_batch = data, lengths, labels
                outputs = F.softmax(self.model(data_batch, length_batch), dim=1)
                predicted = outputs.max(1, keepdim=True)[1]
                cur_loss += F.cross_entropy(outputs, labels).cpu().detach().numpy()

                total += labels.size(0)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
            return 100 * correct / total, cur_loss

    def check_early_stop(self):
        """
        the method called by the standard training loop in BaseModel to determine early stop
        if no early stop is wanted, can just return False
        can also use the hparam to control whether early stop is considered
        :return: bool whether to stop the loop
        """
        val_acc_history = self.iter_curves[self.VAL_ACC]
        t = self.hparams[HyperParamKey.EARLY_STOP_LOOK_BACK]
        required_progress = self.hparams[HyperParamKey.EARLY_STOP_REQ_PROG]

        if len(val_acc_history) >= t + 1 and val_acc_history[-t - 1] > max(val_acc_history[-t:]) - required_progress:
            return True
        return False
