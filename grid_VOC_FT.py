"""
Grid search
Vocab_size and use of pretrained embeddings
"""

import logging
import importlib
import time
import argparse

import libs.common.utils as utils
from libs import ModelManager as mm
from config.constants import PathKey, HyperParamKey


# parse cmd-line parameters
parser = argparse.ArgumentParser(
    description="NLP Team Project - Machine Translation - Grid search")
parser.add_argument('-d', '--DATA', dest='data_path',
                    help='path of data files')
parser.add_argument('-s', '--MSAVE', dest='model_save',
                    help='path of model checkpoints')
parser.add_argument('-c', '--CONFIG', dest='config_file',
                    help='config file name (basic parameters, no tuning here)')
parser.add_argument('-m', '--MODEL', dest='model_type', required=True,
                    help='type of model that you will tune')
parser.add_argument('-v', '--VOCAB', dest='vocab_size', required=True)
parser.add_argument('-u', '--USE_FT', dest='use_ft_emb', required=True)
parser.add_argument('-f', '--FREEZE', dest='freeze', required=False)

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
output_fn = '{}gridSearch-{}{}'.format(
    config_new[PathKey.MODEL_SAVES], args.model_type, ts)
utils.init_logger(logfile=None)
logger = logging.getLogger('__main__')

########################
# Hyper-parameter Lists #
########################

# tune
mgr = mm.ModelManager(hparams=hparam_new, control_overrides=config_new)

hparam_new = {
    HyperParamKey.VOC_SIZE: int(args.vocab_size),  # remember to cast type!
    HyperParamKey.USE_FT_EMB: bool(args.use_ft_emb),
    HyperParamKey.FREEZE_EMB: bool(args.freeze),
}
label = utils.hparam_to_label(prefix=args.model_type, hparam_dict=hparam_new)
mgr.hparams.update(hparam_new)
mgr.load_data(mm.loaderRegister.IWSLT)
mgr.new_model(args.model_type, label=label)
mgr.train()
mgr.graph_training_curves()
mgr.get_results().to_csv(output_fn + '.csv')

logger.info(
    "Single model train complete.\nModel {} {} training report:\n{}\n===\n===\n===".format(
    args.model_type, label, mgr.model.output_dict))
