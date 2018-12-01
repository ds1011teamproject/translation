"""
Basic Seq2Seq (Encoder-Decoder) model for translation.
"""
import logging

from config.basic_conf import DEVICE
from config.constants import HyperParamKey, LoaderParamKey
from libs.data_loaders.IwsltLoader import SRC
from libs.models.RNN_GRU import RNN_GRU
from libs.models.RNN_Attention import RNN_Attention
from libs.models.modules import CNN as cnn

logger = logging.getLogger('__main__')


class CNN_Pool(RNN_GRU):
    def __init__(self, hparams, lparams, cparams,
                 label="cnn-scratch", nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        # Overriding the encoder with CNN
        self.encoder = cnn.EncoderPool(
            vocab_size=self.lparams[LoaderParamKey.ACT_VOCAB_SIZE][SRC],
            emb_size=self.hparams[HyperParamKey.EMBEDDING_DIM],
            hidden_size=self.hparams[HyperParamKey.HIDDEN_SIZE],
            kernel_size=self.hparams[HyperParamKey.KERNEL_SIZE],
            dropout_prob=self.hparams.get(HyperParamKey.ENC_DROPOUT, 0.0),
            trained_emb=lparams[LoaderParamKey.TRAINED_EMB] if hparams[HyperParamKey.USE_FT_EMB] else None,
            freeze_emb=hparams[HyperParamKey.FREEZE_EMB] if hparams[HyperParamKey.USE_FT_EMB] else False
        ).to(DEVICE)


class CNNAttn_Pool(RNN_Attention):
    def __init__(self, hparams, lparams, cparams,
                 label="cnn-scratch", nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        # Overriding the encoder with CNN
        self.encoder = cnn.EncoderPool(
            vocab_size=self.lparams[LoaderParamKey.ACT_VOCAB_SIZE][SRC],
            emb_size=self.hparams[HyperParamKey.EMBEDDING_DIM],
            hidden_size=self.hparams[HyperParamKey.HIDDEN_SIZE],
            kernel_size=self.hparams[HyperParamKey.KERNEL_SIZE],
            dropout_prob=self.hparams.get(HyperParamKey.ENC_DROPOUT, 0.0),
            trained_emb=lparams[LoaderParamKey.TRAINED_EMB] if hparams[HyperParamKey.USE_FT_EMB] else None,
            freeze_emb=hparams[HyperParamKey.FREEZE_EMB] if hparams[HyperParamKey.USE_FT_EMB] else False,
            use_attn=True
        ).to(DEVICE)


class CNN_Tanh(RNN_GRU):
    def __init__(self, hparams, lparams, cparams,
                 label="cnn-scratch", nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        self.encoder = cnn.EncoderTanh(
            vocab_size=self.lparams[LoaderParamKey.ACT_VOCAB_SIZE][SRC],
            emb_size=self.hparams[HyperParamKey.EMBEDDING_DIM],
            hidden_size=self.hparams[HyperParamKey.HIDDEN_SIZE],
            kernel_size=self.hparams[HyperParamKey.KERNEL_SIZE],
            seq_len=self.hparams[HyperParamKey.MAX_LENGTH],
            dropout_prob=self.hparams.get(HyperParamKey.ENC_DROPOUT, 0.0),
            trained_emb=lparams[LoaderParamKey.TRAINED_EMB] if hparams[HyperParamKey.USE_FT_EMB] else None,
            freeze_emb=hparams[HyperParamKey.FREEZE_EMB] if hparams[HyperParamKey.USE_FT_EMB] else False
        ).to(DEVICE)


class CNNAttn_Tanh(RNN_Attention):
    def __init__(self, hparams, lparams, cparams,
                 label="cnn-scratch", nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        self.encoder = cnn.EncoderTanh(
            vocab_size=self.lparams[LoaderParamKey.ACT_VOCAB_SIZE][SRC],
            emb_size=self.hparams[HyperParamKey.EMBEDDING_DIM],
            hidden_size=self.hparams[HyperParamKey.HIDDEN_SIZE],
            kernel_size=self.hparams[HyperParamKey.KERNEL_SIZE],
            seq_len=self.hparams[HyperParamKey.MAX_LENGTH],
            dropout_prob=self.hparams.get(HyperParamKey.ENC_DROPOUT, 0.0),
            trained_emb=lparams[LoaderParamKey.TRAINED_EMB] if hparams[HyperParamKey.USE_FT_EMB] else None,
            freeze_emb=hparams[HyperParamKey.FREEZE_EMB] if hparams[HyperParamKey.USE_FT_EMB] else False,
            use_attn=True
        ).to(DEVICE)


class CNN_Relu(RNN_GRU):
    def __init__(self, hparams, lparams, cparams,
                 label="cnn-scratch", nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        self.encoder = cnn.EncoderRelu(
            vocab_size=self.lparams[LoaderParamKey.ACT_VOCAB_SIZE][SRC],
            emb_size=self.hparams[HyperParamKey.EMBEDDING_DIM],
            hidden_size=self.hparams[HyperParamKey.HIDDEN_SIZE],
            kernel_size=self.hparams[HyperParamKey.KERNEL_SIZE],
            seq_len=self.hparams[HyperParamKey.MAX_LENGTH],
            dropout_prob=self.hparams.get(HyperParamKey.ENC_DROPOUT, 0.0),
            trained_emb=lparams[LoaderParamKey.TRAINED_EMB] if hparams[HyperParamKey.USE_FT_EMB] else None,
            freeze_emb=hparams[HyperParamKey.FREEZE_EMB] if hparams[HyperParamKey.USE_FT_EMB] else False
        ).to(DEVICE)


class CNNAttn_Relu(RNN_Attention):
    def __init__(self, hparams, lparams, cparams,
                 label="cnn-scratch", nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        self.encoder = cnn.EncoderRelu(
            vocab_size=self.lparams[LoaderParamKey.ACT_VOCAB_SIZE][SRC],
            emb_size=self.hparams[HyperParamKey.EMBEDDING_DIM],
            hidden_size=self.hparams[HyperParamKey.HIDDEN_SIZE],
            kernel_size=self.hparams[HyperParamKey.KERNEL_SIZE],
            seq_len=self.hparams[HyperParamKey.MAX_LENGTH],
            dropout_prob=self.hparams.get(HyperParamKey.ENC_DROPOUT, 0.0),
            trained_emb=lparams[LoaderParamKey.TRAINED_EMB] if hparams[HyperParamKey.USE_FT_EMB] else None,
            freeze_emb=hparams[HyperParamKey.FREEZE_EMB] if hparams[HyperParamKey.USE_FT_EMB] else False,
            use_attn=True
        ).to(DEVICE)

