"""
hyperparameter settings, defaults
"""
from config.constants import HyperParamKey
import torch


DEFAULT_HPARAMS = {
    HyperParamKey.NUM_EPOCH: 1,
    HyperParamKey.LR: 0.01,
    HyperParamKey.TRAIN_PLUS_VAL_SIZE: 25000,
    HyperParamKey.TEST_SIZE: 25000,
    HyperParamKey.VAL_SIZE: 5000,
    HyperParamKey.VOC_SIZE: 100000,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 100,
    HyperParamKey.EMBEDDING_DIM: 50,
    HyperParamKey.BATCH_SIZE: 32,
    HyperParamKey.NGRAM_SIZE: 2,
    HyperParamKey.REMOVE_PUNC: True,
    HyperParamKey.CHECK_EARLY_STOP: True,
    HyperParamKey.EARLY_STOP_LOOK_BACK: 5,
    HyperParamKey.EARLY_STOP_REQ_PROG: 0.01,
    HyperParamKey.OPTIMIZER_ENCODER: torch.optim.Adam,
    HyperParamKey.OPTIMIZER_DECODER: torch.optim.Adam,  # not needed, but kept for future use
    HyperParamKey.SCHEDULER: torch.optim.lr_scheduler.ExponentialLR,
    HyperParamKey.SCHEDULER_GAMMA: 0.95,
    HyperParamKey.CRITERION: torch.nn.CrossEntropyLoss
}
