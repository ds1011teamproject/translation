"""
non hyperparameter settings
"""
import logging
import logging.config
import torch
import sys

from config.constants import PathKey, LogConfig


DEFAULT_PATHS = {
    PathKey.INPUT_LANG: 'data/training/europarl-v7.fr-en.en',
    PathKey.OUTPUT_LANG: 'data/training/europarl-v7.fr-en.fr',
    PathKey.ENC_SAVE: 'model_saves/gru_encoder.pkl',
    PathKey.DEC_SAVE: 'model_saves/gru_decoder.pkl',
    PathKey.RESULT_SAVE: 'model_saves/results.p'
}

# todo: need to deprecate this and use the ref in ModelManager
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_LEVEL_DEFAULT = getattr(logging, LogConfig['handlers']['default']['level'])


def init_logger(loglevel=LOG_LEVEL_DEFAULT, logfile='mt.log'):
    logging.getLogger('__main__').setLevel(loglevel)
    LogConfig['handlers']['default']['filename'] = logfile
    logging.config.dictConfig(LogConfig)

