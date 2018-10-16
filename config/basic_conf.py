"""
non hyperparameter settings
"""
import logging
import torch
import sys

from config.constants import PathKey


DEFAULT_PATHS = {
    PathKey.INPUT_LANG: 'data/training/europarl-v7.fr-en.en',
    PathKey.OUTPUT_LANG: 'data/training/europarl-v7.fr-en.fr',
    PathKey.ENC_SAVE: 'model_saves/gru_encoder.pkl',
    PathKey.DEC_SAVE: 'model_saves/gru_decoder.pkl',
    PathKey.RESULT_SAVE: 'model_saves/results.p'
}

LOG_FORMAT = '%(levelname)-8s %(message)s'
LOG_STREAM = sys.stdout
LOG_LEVEL_DEFAULT = logging.DEBUG

# todo: need to deprecate this and use the ref in ModelManager
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_logger(loglevel=LOG_LEVEL_DEFAULT):
    logging.basicConfig(stream=LOG_STREAM, level=loglevel, format=LOG_FORMAT)
