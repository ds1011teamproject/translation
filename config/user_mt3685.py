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

HPARAM = {
    HyperParamKey.EMBEDDING_DIM: 300,
    HyperParamKey.ENC_LR: 0.001,
    HyperParamKey.DEC_LR: 0.001,
    HyperParamKey.NUM_EPOCH: 1,
    HyperParamKey.BATCH_SIZE: 32,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 200,
    HyperParamKey.CHECK_EARLY_STOP: True,
    HyperParamKey.USE_FT_EMB: True,
    HyperParamKey.NUM_TRAIN_SENT_TO_LOAD: None
}
