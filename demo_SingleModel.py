"""
DEMO: Train a single model.

Command-line parameters:
-d: DATA_PATH
-s: MODEL_SAVE
-c: CONFIG FILE NAME (contains paths and hparams)
-m: MODEL_TYPE (RNN_GRU, CNN, or RNN_Attention) Default: RNN_GRU
-l: MODEL_LABEL [required]

Usage:

1. If you use default hyper-parameters, please specify your data_path and model_save path,
   model_label is required:

    $ python demo_SingleModel.py -d data/ -s model_saves/ -m RNN_GRU -l myGRUdemo

2. [Suggested] Put your config file in folder ./config/, and specify your config file name
   (note: file name only, no suffix '.py'), model_label is required. In this way, you can
   change both your data paths and your hyper-parameters:

    $ python demo_SingleModel.py -c user_foo -m RNN_GRU -l myGRUdemo

3. Simplest:

    $ python demo_SingleModel.py -c <your_config> -l GRUdemo

"""

import logging
import importlib
import time
import argparse

import libs.common.utils
from libs import ModelManager as mm
from config.constants import PathKey


# parse cmd-line parameters
parser = argparse.ArgumentParser(description="DS-GA 1011 NLP Team Project - Machine Translation")
parser.add_argument('-d', '--DATA', dest='data_path',
                    help='path of data files')
parser.add_argument('-s', '--MSAVE', dest='model_save',
                    help='path of model checkpoints')
parser.add_argument('-c', '--CONFIG', dest='config_file',
                    help='config file path')
parser.add_argument('-m', '--MODEL', dest='model_type', default='RNN_GRU',
                    help='model type of your choice')
parser.add_argument('-l', '--LABEL', dest='model_label', required=True,
                    help='model label/name')
args = parser.parse_args()

# new config
config_new = {}
hparam_new = None
if getattr(args, 'config_file'):
    user_conf = importlib.import_module('config.{}'.format(args.config_file))
    config_new.update(user_conf.CONFIG)
    hparam_new = user_conf.HPARAM
if getattr(args, 'data_path'):
    config_new.update({PathKey.DATA_PATH: args.data_path})
if getattr(args, 'model_save'):
    config_new.update({PathKey.MODEL_SAVES: args.model_save})

# logger
libs.common.utils.init_logger(logfile='{}{}-{}.log'.format(
    config_new[PathKey.MODEL_SAVES], args.model_label, time.strftime("%m-%d-%H:%M:%S")))
logger = logging.getLogger('__main__')

# model manager
mgr = mm.ModelManager(hparams=hparam_new, control_overrides=config_new)
mgr.load_data(mm.loaderRegister.IWSLT)
mgr.new_model(args.model_type, label=args.model_label)
mgr.train()
logger.info("Demo RNN_GRU report:\n{}".format(mgr.model.output_dict))
