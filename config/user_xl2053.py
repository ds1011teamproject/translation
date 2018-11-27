"""
Xialiang's config file
"""

from config.constants import PathKey, HyperParamKey


CONFIG = {
    PathKey.DATA_PATH: '/scratch/xl2053/nlp/',
    PathKey.INPUT_LANG: 'vi',
    PathKey.OUTPUT_LANG: 'en',
    PathKey.MODEL_SAVES: '/scratch/xl2053/nlp/translation/model_saves/'
}

HPARAM = {
    HyperParamKey.EMBEDDING_DIM: 200,
    HyperParamKey.ENC_LR: 0.001,
    HyperParamKey.DEC_LR: 0.001,
    HyperParamKey.NUM_EPOCH: 1,
    HyperParamKey.ENC_NUM_DIRECTIONS: 2,
    HyperParamKey.DEC_NUM_DIRECTIONS: 1,
    HyperParamKey.BATCH_SIZE: 64,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 200,
    HyperParamKey.NUM_TRAIN_SENT_TO_LOAD: None,
    HyperParamKey.CHECK_EARLY_STOP: True,
    HyperParamKey.USE_FT_EMB: False,
    HyperParamKey.TEACHER_FORCING_RATIO: 1.0
}
