"""
DEMO: Hyper-parameters Grid Search

Command-line parameters:
-d: DATA_PATH
-s: MODEL_SAVE_PATH
-c: CONFIG FILE NAME (in which contains paths and hparams)
-m: MODEL_TYPE (RNN_GRU, CNN, or RNN_Attention) [required]
-l: LR learning rate [required]

Usage:

############################
# Put the hyper-parameters #
# you are going to tune or #
# search in this script.   #
############################

1. If you use default hyper-parameters, please specify your data_path and model_save path,
   model_type is required:

    $ python lr_grid.py -d data/ -s model_saves/ -m RNN_GRU -l 0.01

2. [Suggested] Put your config file in folder ./config/, and specify your config file name
   (NOTE: file name only, no suffix '.py'), model_type is required. In this way, you can
   change both your data paths and your hyper-parameters (those not tuning):

    $ python lr_grid.py -c user_config -m RNN_GRU -l 0.01


Output:

1. Model checkpoints:
    For each combination of hyperparameters, it will save the trained model into folder named

        <model_type>[param1][param1value][param2][param2value]...

        e.g.: RNNGRUusefFalsevocs100000maxl60

2. Result summary:
    By the end of the program, it will generate a csv file which contains the output_dict from
    each model. The output_dict contains best_val_BLEU and final_val_BLEU, etc. The csv file is
    named after:

        gridSearch-<model_type>-%m-%d-%H:%M:%S.csv

        e.g. gridSearch-RNN_GRU11-28-16:43:31.csv

3. Log file:
    Log file is saved in MODEL_SAVES directory named as

        gridSearch-<model_type>-%m-%d-%H:%M:%S.log

        e.g. gridSearch-RNN_GRU11-28-16:43:31.log

"""

import logging
import importlib
import time
import argparse

import libs.common.utils as utils
from libs import ModelManager as mm
from config.constants import PathKey, HyperParamKey


# parse cmd-line parameters
parser = argparse.ArgumentParser(description="NLP Team Project - Machine Translation - Grid search")
parser.add_argument('-d', '--DATA', dest='data_path',
                    help='path of data files')
parser.add_argument('-s', '--MSAVE', dest='model_save',
                    help='path of model checkpoints')
parser.add_argument('-c', '--CONFIG', dest='config_file',
                    help='config file name (contains basic parameters, normally not tuning here)')
parser.add_argument('-m', '--MODEL', dest='model_type', required=True,
                    help='type of model that you will tune')
parser.add_argument('-l', '--learning-rate', dest='lr', required=True,
                    help='the learning rate to search')
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
ts = time.strftime("%m-%d-%H:%M:%S")
output_fn = '{}gridSearch-{}{}'.format(config_new[PathKey.MODEL_SAVES], args.model_type, ts)
utils.init_logger(loglevel=logging.INFO, logfile=output_fn + '.log')
logger = logging.getLogger('__main__')

########################
# Hyper-parameter Lists #
########################

# tune
mgr = mm.ModelManager(hparams=hparam_new, control_overrides=config_new)

hparam_new = {
    HyperParamKey.ENC_LR: float(args.lr),  # remember to cast to right type!
    HyperParamKey.DEC_LR: float(args.lr),  # remember to cast to right type!
}
label = utils.hparam_to_label(prefix=args.model_type, hparam_dict=hparam_new)
mgr.hparams.update(hparam_new)
mgr.load_data(mm.loaderRegister.IWSLT)
mgr.new_model(args.model_type, label=label)
mgr.train()
mgr.graph_training_curves()
mgr.get_results().to_csv(output_fn + '.csv')

