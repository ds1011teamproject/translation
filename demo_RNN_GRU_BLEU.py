"""
RNN(GRU) model demo
"""

import logging
from tqdm import tqdm

import libs.common.utils
from config import basic_conf as conf
from config.constants import PathKey, HyperParamKey

from libs import ModelManager as mm
from libs.common.BleuScorer import BLEUScorer
import libs.data_loaders.IwsltLoader as iwslt
from libs.data_loaders.IwsltLoader import DataSplitType


# logger
libs.common.utils.init_logger(logging.INFO, logfile='fasttext.log')
logger = logging.getLogger('__main__')

# ==== CHANGE YOUR DATA_PATH, MODEL_SAVES ====
config_new = {
    PathKey.DATA_PATH: '/scratch/mt3685/translation/',
    PathKey.INPUT_LANG: 'vi',
    PathKey.OUTPUT_LANG: 'en',
    PathKey.MODEL_SAVES: '/scratch/mt3685/translation/model_saves/'
}
hparam_new = {
    HyperParamKey.EMBEDDING_DIM: 300,
    HyperParamKey.ENC_LR: 0.001,
    HyperParamKey.DEC_LR: 0.001,
    HyperParamKey.NUM_EPOCH: 1,
    HyperParamKey.BATCH_SIZE: 32,
    HyperParamKey.TRAIN_LOOP_EVAL_FREQ: 200,
    HyperParamKey.CHECK_EARLY_STOP: False,
    HyperParamKey.USE_FT_EMB: True,
    HyperParamKey.NUM_TRAIN_SENT_TO_LOAD: None,
}

# Train new model
mgr = mm.ModelManager(hparams=hparam_new, control_overrides=config_new)
mgr.load_data(mm.loaderRegister.IWSLT)
mgr.new_model(mm.modelRegister.RNN_GRU, label='fasttext_test')
mgr.train()
logger.info("Demo RNN_GRU_BLEU report:\n{}".format(mgr.model.output_dict))

# Load trained model
mgr.model = None
mgr.new_model(mm.modelRegister.RNN_GRU, label='fasttext_test', nolog=True)
mgr.load_model(which_model='checkpoint.tar')

# Translate validation set
true = []
pred = []

loader = mgr.dataloader
id2token = loader.id2token[iwslt.TAR]

# Iterate over batches
for src, tgt, slen, _ in tqdm(loader.loaders[DataSplitType.VAL]):
    # print(len(src))  # 32

    # Iterate over sentences
    for idx in range(len(src)):
        # print(src.shape)  # torch.Size([1, 100])

        # Convert single sentence to look like a batch of 1 sample
        src_ = src[idx].unsqueeze(0)
        tgt_ = tgt[idx].unsqueeze(0)
        slen_ = slen[idx].unsqueeze(0)

        mgr.model.encoder.eval()
        mgr.model.decoder.eval()

        # Encoding
        enc_hidden = mgr.model.encoder(src_, slen_)
        # Decoding
        predicted = mgr.model.decoding(tgt_, enc_hidden,
                                       teacher_forcing=False, mode="translate")
        # Convert to strings
        target = " ".join([
            id2token[e] for e in tgt_.squeeze() if e != iwslt.PAD_IDX])
        translated = " ".join([id2token[e] for e in predicted])
        true.append(target)
        pred.append(translated)

# Compute BLEU score for the validation set
scorer = BLEUScorer()
assert(len(true) == len(pred))

result = scorer.bleu(true, [pred], score_only=False)
print(result)
