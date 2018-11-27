"""
non hyperparameter settings
"""
import torch
from config.constants import PathKey, ControlKey

DEFAULT_CONTROLS = {
    ControlKey.SAVE_BEST_MODEL: True,
    ControlKey.SAVE_EACH_EPOCH: True,
    PathKey.INPUT_LANG: 'zh',
    PathKey.DATA_PATH: 'data/',
    PathKey.OUTPUT_LANG: 'en',
    PathKey.MODEL_SAVES: 'model_saves/'
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


