from config import basic_conf as conf
from libs import ModelManager as mm
from config.constants import PathKey
import logging

conf.init_logger(logging.INFO)
logger = logging.getLogger('__main__')


config_new = {
    PathKey.DATA_PATH: '/Users/wyz0214/Downloads/',
    PathKey.INPUT_LANG: 'zh',
    PathKey.OUTPUT_LANG: 'en'
}

mgr = mm.ModelManager(control_overrides=config_new)

mgr.load_data(mm.loaderRegister.IWSLT)

mgr.new_model(mm.modelRegister.RNN_GRUattn, label='GRUAttn trial')

mgr.train()
