"""
Melanie's config file
"""

from config.constants import PathKey, HyperParamKey


# Please DON'T change the name of these 2 dicts: CONFIG and HPARAM
CONFIG = {
    PathKey.DATA_PATH: '/scratch/mt3685/translation/',
    PathKey.INPUT_LANG: 'vi',
    PathKey.OUTPUT_LANG: 'en',
    PathKey.MODEL_SAVES: '/scratch/mt3685/translation/model_saves/'
}

HPARAM = {}

