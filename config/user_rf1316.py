"""
Rong's config file
(Use your local path by default)

"""
from config.constants import PathKey, HyperParamKey


# Please DON'T change the name of these 2 dicts: CONFIG and HPARAM
CONFIG = {
    PathKey.DATA_PATH: 'data/',
    PathKey.MODEL_SAVES: 'model_saves/',
    PathKey.INPUT_LANG: 'zh',
    PathKey.OUTPUT_LANG: 'en'
}

HPARAM = {
    HyperParamKey.EMBEDDING_DIM: 300,
    HyperParamKey.ENC_LR: 0.01,
    HyperParamKey.DEC_LR: 0.01,
    HyperParamKey.NUM_EPOCH: 10,
    HyperParamKey.ENC_NUM_DIRECTIONS: 1,
    HyperParamKey.DEC_NUM_DIRECTIONS: 1,
    HyperParamKey.BATCH_SIZE: 32,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 200,
    HyperParamKey.NUM_TRAIN_SENT_TO_LOAD: None,
    HyperParamKey.CHECK_EARLY_STOP: True,
    HyperParamKey.USE_FT_EMB: False,
    HyperParamKey.TEACHER_FORCING_RATIO: 1.0
}
