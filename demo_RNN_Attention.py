"""
RNN(GRU) with attention demo.
"""

import logging
import sys
import time

from libs import ModelManager as mm
from config import basic_conf as conf
from config.constants import PathKey, HyperParamKey


# ==== CHANGE YOUR LOG_FILE, DATA_PATH, MODEL_SAVES ====
data_path = '/scratch/xl2053/nlp/'
model_save = data_path + 'translation/model_saves/'
if sys.platform == 'linux':  # hpc
    pass
elif sys.platform == 'darwin':  # local test
    data_path = '/Users/xliu/Downloads/'
    model_save = 'model_saves/'

# logger
label = 'gruAttnFullTS8ep'
ts = time.gmtime()
conf.init_logger(loglevel=logging.INFO,
                 logfile='{p}translation/logs/{lb}-{m}-{d}-{H}:{M}.log'.format(
                     p=data_path, lb=label, m=ts[1], d=ts[2], H=ts[3], M=ts[4]))
logger = logging.getLogger('__main__')

# new config
config_new = {
    PathKey.DATA_PATH: data_path,
    PathKey.INPUT_LANG: 'vi',
    PathKey.OUTPUT_LANG: 'en',
    PathKey.MODEL_SAVES: model_save
}
hparam_new = {
    HyperParamKey.EMBEDDING_DIM: 200,
    HyperParamKey.ENC_LR: 0.001,
    HyperParamKey.DEC_LR: 0.001,
    HyperParamKey.NUM_EPOCH: 8,
    HyperParamKey.ENC_NUM_DIRECTIONS: 2,
    HyperParamKey.DEC_NUM_DIRECTIONS: 1,
    HyperParamKey.BATCH_SIZE: 128,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 200,
    HyperParamKey.NUM_TRAIN_SENT_TO_LOAD: None,
    HyperParamKey.CHECK_EARLY_STOP: False,
    HyperParamKey.USE_FT_EMB: False,
    HyperParamKey.TEACHER_FORCING_RATIO: 1.0
}

# model manager
mgr = mm.ModelManager(hparams=hparam_new, control_overrides=config_new)
mgr.load_data(mm.loaderRegister.IWSLT)
mgr.new_model(mm.modelRegister.RNN_Attention, label='gruAttnFullTS1ep')
mgr.train()
logger.info("Demo RNN_Attention report:\n{}".format(mgr.model.output_dict))

