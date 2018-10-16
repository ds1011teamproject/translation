"""
hyperparameter settings, defaults
"""
from config.constants import HyperParamKey


DEFAULT_HPARAMS = {
    HyperParamKey.EMB_SIZE: 500,
    HyperParamKey.HIDDEN_SIZE: 1000,
    HyperParamKey.MAX_LEN: 20,
    HyperParamKey.NUM_BATCH: 7500,
    HyperParamKey.NUM_EPOCH: 1000,
    HyperParamKey.VOC_SIZE: 15000,
}
