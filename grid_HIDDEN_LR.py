"""
Grid search
hidden_size, learning_rate

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
parser.add_argument('-v', '--HIDDEN', dest='hidden_size', required=True)
parser.add_argument('-l', '--LR', dest='learning_rate')
args = parser.parse_args()

# new config
config_new = {}
hparam_new = {}
if getattr(args, 'config_file'):
    user_conf = importlib.import_module('config.{}'.format(args.config_file))
    config_new.update(user_conf.CONFIG)
    hparam_new = user_conf.get('HPARAM', {})
if getattr(args, 'data_path'):
    config_new.update({PathKey.DATA_PATH: args.data_path})
if getattr(args, 'model_save'):
    config_new.update({PathKey.MODEL_SAVES: args.model_save})
if getattr(args, 'learning_rate'):
    hparam_new[HyperParamKey.ENC_LR] = float(args.learning_rate)
    hparam_new[HyperParamKey.DEC_LR] = float(args.learning_rate)
else:
    hparam_new[HyperParamKey.DEC_LR] = 1/float(args.hidden_size)
    hparam_new[HyperParamKey.ENC_LR] = 1/float(args.hidden_size)

# logger
ts = time.strftime("%m-%d-%H:%M:%S")
# output_fn = '{}gridSearch-{}{}'.format(config_new[PathKey.MODEL_SAVES], args.model_type, ts)
utils.init_logger(logfile=None)
logger = logging.getLogger('__main__')

########################
# Hyper-parameter Lists #
########################



# tune
mgr = mm.ModelManager(hparams=hparam_new, control_overrides=config_new)

hparam_new.update({
    HyperParamKey.HIDDEN_SIZE: int(args.hidden_size),  # remember to cast to right type!
})
label = utils.hparam_to_label(prefix=args.model_type, hparam_dict=hparam_new)
mgr.hparams.update(hparam_new)
mgr.load_data(mm.loaderRegister.IWSLT)
mgr.new_model(args.model_type, label=label)
mgr.train()
mgr.graph_training_curves()
# mgr.get_results().to_csv(output_fn + '.csv')

logger.info("Single model train complete.\nModel {} {} training report:\n{}\n===\n===\n===".format(
    args.model_type, label, mgr.model.output_dict))
