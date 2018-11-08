"""
super class for all of the various model we will build

Notes:
- the model should implement its own training loop / early stop
- it should track its own training curve
- it should track its own results

"""
import logging
import torch

from libs.models.BaseModel import BaseModel
from config.constants import (HyperParamKey, LoaderParamKey, ControlKey,
                              PathKey, StateKey, LoadingKey, OutputKey)

logger = logging.getLogger('__main__')


class MTBaseModel(BaseModel):
    def __init__(self, hparams, lparams, cparams, label='scratch', nolog=False):
        super().__init__(hparams, lparams, cparams, label, nolog)

        self.encoder = None
        self.decoder = None
        self.enc_optim = None
        self.dec_optim = None
        self.enc_scheduler = None
        self.dec_scheduler = None

        # todo: might add other metrics later
        self.iter_curves = {
            # self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            # self.VAL_ACC: [],
            self.VAL_LOSS: [],
        }
        # todo
        self.epoch_curves = {
            # self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            # self.VAL_ACC: [],
            self.VAL_LOSS: [],
        }

    def train(self, loader, tqdm_handler):
        # todo: implement, if works for all nmt models; or to override in child model
        # For saving train history (train/validation losses), please refer to
        # the train loop in ClassifierModel.py
        pass

    def save(self, fn=BaseModel.CHECKPOINT_FN):
        state = {
            StateKey.MODEL_STATE: {'encoder': self.encoder.state_dict(),
                                   'decoder': self.decoder.state_dict()},
            StateKey.OPTIM_STATE: {'encoder': self.enc_optim.state_dict(),
                                   'decoder': self.dec_optim.state_dict()},
            StateKey.SCHED_STATE: {'encoder': self.enc_scheduler.state_dict(),
                                   'decoder': self.dec_scheduler.state_dict()},
            StateKey.HPARAMS: self.hparams,
            StateKey.CPARAMS: self.cparams,
            StateKey.LPARAMS: self.lparams,
            StateKey.ITER_CURVES: self.iter_curves,
            StateKey.EPOCH_CURVES: self.epoch_curves,
            StateKey.CUR_EPOCH: self.cur_epoch,
            StateKey.LABEL: self.label
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
                path_to_model_ovrd = self.cparams[PathKey.MODEL_PATH] + self.BEST_FN
            else:
                path_to_model_ovrd = self.cparams[PathKey.MODEL_PATH] + self.CHECKPOINT_FN

        logger.info("loading checkpoint at {}".format(path_to_model_ovrd))
        loaded = torch.load(path_to_model_ovrd)

        # load encoder/decoder
        self.encoder.load_state_dict(loaded[StateKey.MODEL_STATE]['encoder'])
        self.decoder.load_state_dict(loaded[StateKey.MODEL_STATE]['decoder'])
        # load optimizers
        self._init_optim()
        self.enc_optim.load_state_dict(loaded[StateKey.OPTIM_STATE]['encoder'])
        self.dec_optim.load_state_dict(loaded[StateKey.OPTIM_STATE]['decoder'])
        # load lr_schedulers
        self._init_scheduler()
        self.enc_scheduler.load_state_dict(loaded[StateKey.SCHED_STATE]['encoder'])
        self.dec_scheduler.load_state_dict(loaded[StateKey.SCHED_STATE]['decoder'])
        # load parameters
        self.hparams = loaded[StateKey.HPARAMS]
        self.lparams = loaded[StateKey.LPARAMS]
        self.cparams = loaded[StateKey.CPARAMS]
        # load train history
        self.iter_curves = loaded[StateKey.ITER_CURVES]
        self.epoch_curves = loaded[StateKey.EPOCH_CURVES]
        self.cur_epoch = loaded[StateKey.CUR_EPOCH]
        self.label = loaded[StateKey.LABEL]

        logger.info("Successfully loaded checkpoint!")

    def _init_optim(self):
        op_constr = self.hparams[HyperParamKey.OPTIMIZER]
        self.enc_optim = op_constr(self.encoder.parameters(), lr=self.hparams[HyperParamKey.ENC_LR])
        self.dec_optim = op_constr(self.decoder.parameters(), lr=self.hparams[HyperParamKey.DEC_LR])

    def _init_scheduler(self):
        sche_constr = self.hparams[HyperParamKey.SCHEDULER]
        self.enc_scheduler = sche_constr(self.enc_optim, gamma=self.hparams[HyperParamKey.SCHEDULER_GAMMA])
        self.dec_scheduler = sche_constr(self.dec_scheduler, gamma=self.hparams[HyperParamKey.SCHEDULER_GAMMA])

    ################################
    # Override following if needed #
    ################################

    def check_early_stop(self):
        pass

    def eval_model(self, dataloader):
        pass
