"""
Basic Seq2Seq (Encoder-Decoder) model for translation.
"""
import logging

from config.basic_conf import DEVICE
from config.constants import (
    HyperParamKey as hparamKey,
    LoaderParamKey as loaderKey,
)
from libs.data_loaders.IwsltLoader import SRC
from libs.models.RNN_GRU import RNN_GRU
from libs.models.modules import CNN_encoder

logger = logging.getLogger('__main__')


class CNN(RNN_GRU):
    def __init__(self, hparams, lparams, cparams,
                 label="cnn-scratch", nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        # Overriding the encoder with CNN
        self.encoder = CNN_encoder.Encoder(
            vocab_size=self.lparams[loaderKey.ACT_VOCAB_SIZE][SRC],
            emb_size=self.hparams[hparamKey.EMBEDDING_DIM],
            hidden_size=self.hparams[hparamKey.HIDDEN_SIZE],
            kernel_size=self.hparams[hparamKey.KERNEL_SIZE],
            dropout_prob=self.hparams.get(hparamKey.ENC_DROPOUT, 0.0),
        ).to(DEVICE)
