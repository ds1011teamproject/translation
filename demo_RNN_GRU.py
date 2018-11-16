"""
RNN(GRU) model demo.
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
hparam_new = {
    HyperParamKey.EMBEDDING_DIM: 300,
    HyperParamKey.ENC_LR: 0.001,
    HyperParamKey.DEC_LR: 0.001,
    HyperParamKey.NUM_EPOCH: 1,
    HyperParamKey.BATCH_SIZE: 32,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 200,
    HyperParamKey.CHECK_EARLY_STOP: False,
    HyperParamKey.USE_FT_EMB: False
}

# model manager
mgr = mm.ModelManager(hparams=hparam_new, control_overrides=config_new)
mgr.load_data(mm.loaderRegister.IWSLT)
mgr.new_model(mm.modelRegister.RNN_GRU, label='gru_test')
mgr.train()
logger.info("Demo RNN_GRU report:\n{}".format(mgr.model.output_dict))

