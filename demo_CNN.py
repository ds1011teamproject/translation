from config import basic_conf as conf
from libs import ModelManager as mm
from config.constants import PathKey
import logging

conf.init_logger(logging.INFO)
logger = logging.getLogger('__main__')


config_new = {
    PathKey.DATA_PATH: '/Users/melanietosik/Downloads/translation/',
    PathKey.INPUT_LANG: 'zh',
    PathKey.OUTPUT_LANG: 'en'
}

# hparam_new = {
#     HyperParamKey.EMBEDDING_DIM: 200,
#     HyperParamKey.ENC_LR: 0.005,
#     HyperParamKey.DEC_LR: 0.005,
#     HyperParamKey.BATCH_SIZE: 128,
#     HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 10,
#     HyperParamKey.CHECK_EARLY_STOP: True,
#     HyperParamKey.KERNEL_SIZE: 3,
# }

mgr = mm.ModelManager(control_overrides=config_new)

mgr.load_data(mm.loaderRegister.IWSLT)

mgr.new_model(mm.modelRegister.CNN, label='cnn_trial')

mgr.train()
