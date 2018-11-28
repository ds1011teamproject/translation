"""
DEMO: Evaluate BLEU score on trained model (load and eval).

Command-line parameters:
-p CHECKPOINT_PATH
-m MODEL_TYPE [RNN_GRU, CNN, RNN_Attention]
-s CHECKPOINT_SAVE_METHOD [best, checkpoint] Default: checkpoint
--beam USE BEAM SEARCH if set
-w BEAM_WIDTH default = 3


Usage:
1. Simple

    $ python demo_TrainedModelBLEU.py -p /scratch/xl2053/nlp/translation/model_saves/attn/ -m RNN_Attention -s best

2. Use beam-search:

    $ python demo_TrainedModelBLEU.py -p /scratch/xl2053/nlp/translation/model_saves/gruFullTS1ep/ -m RNN_GRU -s best --beam -w 3
"""

import logging
import time
import argparse

import libs.common.utils
from libs import ModelManager as mm
from config.constants import LoadingKey


# parse cmd-line parameters
parser = argparse.ArgumentParser(description="Evaluate BLEU score on trained model.")
parser.add_argument('-p', '--PATH', dest='ckp_path', required=True,
                    help='path of checkpoint files')
parser.add_argument('-m', '--MODEL_TYPE', dest='model_type', required=True,
                    help='type of translation model, from (RNN_GRU, CNN, RNN_Attention)')
parser.add_argument('-s', '--SAVE_METHOD', dest='save_method', default=LoadingKey.LOAD_CHECKPOINT,
                    help='checkpoint save method, best or checkpoint')
parser.add_argument('--beam', action='store_true',
                    help='True if use beam search, else False')
parser.add_argument('-w', '--width', dest='beam_width',
                    help='search width for beam search, if applicable')
args = parser.parse_args()

# logger
libs.common.utils.init_logger(
    logfile='{}evalBLEU-{}.log'.format(args.ckp_path, time.strftime("%m-%d-%H:%M:%S")))
logger = logging.getLogger('__main__')

# model manager
mgr = mm.ModelManager()
mgr.load_model(which_model=args.save_method,
               model_type=args.model_type,
               path_to_model_ovrd=args.ckp_path)
mgr.load_data(mm.loaderRegister.IWSLT)
# compute BLEU
beam_width=int(args.beam_width) if args.beam_width else 3
bleu_result = mgr.model.eval_model(mgr.dataloader, score_only=False, use_beam=args.beam, beam_width=beam_width)
logger.info("BLEU Score report:\nModel: {}{}\n{}".format(args.ckp_path, args.save_method, bleu_result))
