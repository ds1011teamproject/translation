"""
hyperparameter settings, defaults
"""
from config.constants import HyperParamKey
import torch


DEFAULT_HPARAMS = {
    # vocab
    HyperParamKey.VOC_SIZE: 100000,
    # model
    HyperParamKey.EMBEDDING_DIM: 50,
    HyperParamKey.HIDDEN_SIZE: 100,
    HyperParamKey.ENC_NUM_LAYERS: 1,
    HyperParamKey.ENC_NUM_DIRECTIONS: 1,
    HyperParamKey.DEC_NUM_LAYERS: 1,
    HyperParamKey.DEC_NUM_DIRECTIONS: 1,
    # train
    HyperParamKey.TEACHER_FORCING_RATIO: 0.5,
    HyperParamKey.NUM_EPOCH: 1,
    HyperParamKey.ENC_LR: 0.01,
    HyperParamKey.DEC_LR: 0.01,
    HyperParamKey.BATCH_SIZE: 32,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 100,
    HyperParamKey.CHECK_EARLY_STOP: True,
    HyperParamKey.EARLY_STOP_LOOK_BACK: 5,
    HyperParamKey.EARLY_STOP_REQ_PROG: 0.01,
    HyperParamKey.OPTIMIZER: torch.optim.Adam,
    HyperParamKey.SCHEDULER: torch.optim.lr_scheduler.ExponentialLR,
    HyperParamKey.SCHEDULER_GAMMA: 0.95,
    HyperParamKey.CRITERION: torch.nn.CrossEntropyLoss
}
