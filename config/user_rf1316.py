"""
Rong's config file
(Use your local path by default)

"""
from config.constants import PathKey, HyperParamKey


# Please DON'T change the name of these 2 dicts: CONFIG and HPARAM
CONFIG = {
    PathKey.DATA_PATH: '/scratch/rf1316/data/',
    PathKey.MODEL_SAVES: '/scratch/rf1316/model_saves/',
    PathKey.INPUT_LANG: 'zh',
    PathKey.OUTPUT_LANG: 'en'
}

HPARAM = {}
