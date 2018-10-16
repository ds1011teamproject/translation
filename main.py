"""
Entry point for the program, argparse
"""

import os
import argparse
import logging

from libs import ModelManager as mm
from config import basic_conf as conf
from config.constants import PathKey, HyperParamKey


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
data_path = args.data_path if getattr(args, 'data_path') else None
config_path = args.config_path if getattr(args, 'config_path') else None

# update parameters to override
paths_new = dict()
if data_path:
    paths_new[PathKey.INPUT_LANG] = os.path.join(data_path, 'europarl-v7.fr-en.en')
    paths_new[PathKey.OUTPUT_LANG] = os.path.join(data_path, 'europarl-v7.fr-en.fr')
logger.info("Paths to override: {}".format(paths_new))

# todo: user input config file format TBD

# todo --- MAIN HERE ---
mgr = mm.ModelManager(hparams=None, path_overrides=paths_new)
mgr.load_data()
mgr.set_model('GRU')
mgr.train()




