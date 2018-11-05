"""
non hyperparameter settings
"""
import logging
import logging.config
import torch
from config.constants import PathKey, LogConfig, ControlKey

DEFAULT_CONTROLS = {
    ControlKey.SAVE_BEST_MODEL: True,
    ControlKey.SAVE_EACH_EPOCH: True,
    PathKey.DATA_PATH_VI: 'data/iwslt-vi-en/',
    PathKey.DATA_PATH_ZH: 'data/iwslt-zh-en/',
    PathKey.INPUT_LANG: 'vi',
    PathKey.OUTPUT_LANG: 'en',
    PathKey.MODEL_SAVES: 'model_saves/'
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_LEVEL_DEFAULT = getattr(logging, LogConfig['handlers']['default']['level'])


def init_logger(loglevel=LOG_LEVEL_DEFAULT, logfile=None):
    logging.getLogger('__main__').setLevel(loglevel)
    if logfile is None:
        LogConfig['loggers']['']['handlers'] = ['console']
        LogConfig['handlers']['default']['filename'] = 'mt.log'
    else:
        LogConfig['loggers']['']['handlers'] = ['console', 'default']
        LogConfig['handlers']['default']['filename'] = logfile
    logging.config.dictConfig(LogConfig)

