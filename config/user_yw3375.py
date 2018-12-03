"""
Ethan's config file
"""

from config.constants import PathKey, HyperParamKey


# Please DON'T change the name of these 2 dicts: CONFIG and HPARAM
CONFIG = {
    PathKey.DATA_PATH: '<folder>',
    PathKey.INPUT_LANG: 'vi',
    PathKey.OUTPUT_LANG: 'en',
    PathKey.MODEL_SAVES: '<folder>/model_saves/'
}

HPARAM = {}
