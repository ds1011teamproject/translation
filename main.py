"""
Entry point for the program, argparse
"""

import os
import argparse
import logging.config

from libs import ModelManager as mm
from config import basic_conf as conf
from config.constants import PathKey


# logger
conf.init_logger()
logger = logging.getLogger('__main__')

# parse cmd-line parameters
parser = argparse.ArgumentParser(description="DS-GA 1011 NLP Team Project - Machine Translation")
parser.add_argument('-p', '--data', dest='data_path',
                    help='path of data files')
parser.add_argument('-c', '--config', dest='config_path',
                    help='path to configuration files')
args = parser.parse_args()
# todo: add kwarg overrides for hparams

data_path = args.data_path if getattr(args, 'data_path') else None
config_path = args.config_path if getattr(args, 'config_path') else None

# update parameters to override
paths_new = dict()
if data_path:
    paths_new[PathKey.TEST_PATH] = os.path.join(data_path, 'aclImdb/test/')
    paths_new[PathKey.TRAIN_PATH] = os.path.join(data_path, 'aclImdb/train/')
logger.info("Paths to override: {}".format(paths_new))

# todo: user input config file format TBD

# List implemented models
logger.info("Implemented models: {}".format(mm.modelRegister.model_list))

# todo --- MAIN HERE ---
mgr = mm.ModelManager(hparams=None, control_overrides=paths_new)
mgr.load_data(mm.loaderRegister.IMDB)
mgr.new_model(mm.modelRegister.BagOfWords)
mgr.train()
mgr.graph_training_curves()

