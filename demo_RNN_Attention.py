"""
RNN(GRU) with attention demo.
"""

import logging

from libs import ModelManager as mm
from config import basic_conf as conf
from config.constants import PathKey, HyperParamKey


# logger
conf.init_logger()
logger = logging.getLogger('__main__')

# ==== CHANGE YOUR DATA_PATH, MODEL_SAVES ====
config_new = {
    PathKey.DATA_PATH: '/scratch/xl2053/nlp/',
    PathKey.INPUT_LANG: 'vi',
    PathKey.OUTPUT_LANG: 'en',
    PathKey.MODEL_SAVES: '/scratch/xl2053/nlp/translation/model_saves/'
}
# ==== My local test =====
# config_new.update({
#     PathKey.DATA_PATH: '/Users/xliu/Downloads/',
#     PathKey.MODEL_SAVES: 'model_saves/'
# })
hparam_new = {
    HyperParamKey.EMBEDDING_DIM: 300,
    HyperParamKey.ENC_LR: 0.001,
    HyperParamKey.DEC_LR: 0.001,
    HyperParamKey.NUM_EPOCH: 3,
    HyperParamKey.ENC_NUM_DIRECTIONS: 2,
    HyperParamKey.DEC_NUM_DIRECTIONS: 1,
    HyperParamKey.BATCH_SIZE: 16,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 200,
    HyperParamKey.NUM_TRAIN_SENT_TO_LOAD: None,
    HyperParamKey.CHECK_EARLY_STOP: False,
    HyperParamKey.USE_FT_EMB: False
}

# model manager
mgr = mm.ModelManager(hparams=hparam_new, control_overrides=config_new)
mgr.load_data(mm.loaderRegister.IWSLT)
mgr.new_model(mm.modelRegister.RNN_Attention, label='attn')
mgr.train()
logger.info("Demo RNN_Attention report:\n{}".format(mgr.model.output_dict))

