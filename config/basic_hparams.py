"""
hyperparameter settings, defaults
"""
from config.constants import HyperParamKey
import torch


DEFAULT_HPARAMS = {
    # vocab
    HyperParamKey.VOC_SIZE: 100000,
    HyperParamKey.USE_FT_EMB: False,
    HyperParamKey.FREEZE_EMB: False,
    # model
    HyperParamKey.EMBEDDING_DIM: 300,
    HyperParamKey.HIDDEN_SIZE: 1000,
    HyperParamKey.ENC_NUM_LAYERS: 1,
    HyperParamKey.ENC_NUM_DIRECTIONS: 1,
    HyperParamKey.DEC_NUM_LAYERS: 1,
    HyperParamKey.DEC_NUM_DIRECTIONS: 1,
    HyperParamKey.KERNEL_SIZE: 3,
    HyperParamKey.MAX_LENGTH: 100,
    # train
    HyperParamKey.TEACHER_FORCING_RATIO: 1.0,  # full teacher forcing
    HyperParamKey.BEAM_SEARCH_WIDTH: 3,
    HyperParamKey.NUM_EPOCH: 100,
    HyperParamKey.ENC_LR: 0.01,
    HyperParamKey.DEC_LR: 0.01,
    HyperParamKey.BATCH_SIZE: 32,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 100,
    HyperParamKey.CHECK_EARLY_STOP: True,
    HyperParamKey.EARLY_STOP_LOOK_BACK: 30,
    HyperParamKey.EARLY_STOP_REQ_PROG: 0.01,
    HyperParamKey.NO_IMPROV_LOOK_BACK: 20,
    HyperParamKey.NO_IMPROV_LR_DECAY: 0.5,
    HyperParamKey.OPTIMIZER: torch.optim.Adam,
    HyperParamKey.SCHEDULER: torch.optim.lr_scheduler.ExponentialLR,
    HyperParamKey.SCHEDULER_GAMMA: 0.95,
    HyperParamKey.CRITERION: torch.nn.functional.nll_loss,
    # testing implementation parameters
    HyperParamKey.NUM_TRAIN_SENT_TO_LOAD: -1,  # -1 or None indicates to load everything
}
