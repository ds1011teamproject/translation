"""
Basic Seq2Seq (Encoder-Decoder) model for translation.
"""
import logging
import torch

from libs.models.TranslatorModel import MTBaseModel, DecodeMode
from libs.models.modules import GRU
import libs.data_loaders.IwsltLoader as iwslt
from config.constants import HyperParamKey, LoaderParamKey
from config.basic_conf import DEVICE
from libs.common.BeamSearcher import beam_search

logger = logging.getLogger('__main__')


class RNN_GRU(MTBaseModel):

    def __init__(self, hparams, lparams, cparams, label='gru-scratch', nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        # encoder-decoder system
        self.encoder = GRU.Encoder(vocab_size=lparams[LoaderParamKey.ACT_VOCAB_SIZE][iwslt.SRC],
                                   emb_size=hparams[HyperParamKey.EMBEDDING_DIM],
                                   hidden_size=hparams[HyperParamKey.HIDDEN_SIZE],
                                   num_layers=hparams[HyperParamKey.ENC_NUM_LAYERS],
                                   num_directions=hparams[HyperParamKey.ENC_NUM_DIRECTIONS],
                                   dropout_prob=hparams.get(HyperParamKey.ENC_DROPOUT, 0.0),
                                   trained_emb=lparams[LoaderParamKey.TRAINED_EMB][iwslt.SRC] if hparams[HyperParamKey.USE_FT_EMB] else None,
                                   freeze_emb=hparams[HyperParamKey.FREEZE_EMB] if hparams[HyperParamKey.USE_FT_EMB] else False
                                   ).to(DEVICE)
        self.decoder = GRU.Decoder(vocab_size=lparams[LoaderParamKey.ACT_VOCAB_SIZE][iwslt.TAR],
                                   emb_size=hparams[HyperParamKey.EMBEDDING_DIM],
                                   hidden_size=hparams[HyperParamKey.HIDDEN_SIZE],
                                   num_layers=hparams[HyperParamKey.DEC_NUM_LAYERS],
                                   num_directions=hparams[HyperParamKey.DEC_NUM_DIRECTIONS],
                                   dropout_prob=hparams.get(HyperParamKey.DEC_DROPOUT, 0.0),
                                   trained_emb=lparams[LoaderParamKey.TRAINED_EMB][iwslt.TAR] if hparams[HyperParamKey.USE_FT_EMB] else None,
                                   freeze_emb=hparams[HyperParamKey.FREEZE_EMB] if hparams[HyperParamKey.USE_FT_EMB] else False
                                   ).to(DEVICE)

    def decoding(self, tgt_batch, enc_results, teacher_forcing, mode, beam_width=3):
        # init
        batch_size = tgt_batch.size(0)
        batch_loss = 0
        predicted = []
        # first input
        dec_in = torch.LongTensor([iwslt.SOS_IDX] * batch_size).unsqueeze(1).to(DEVICE)
        # first hidden
        hidden = self.decoder.init_hidden(enc_results)
        # decoding - beamsearch
        if mode == DecodeMode.TRANSLATE_BEAM:
            # implement beam search here
            beam_width = self.hparams.get(HyperParamKey.BEAM_SEARCH_WIDTH, beam_width)
            with torch.no_grad():
                predicted = beam_search(dec_in, hidden, enc_results, self.decoder, tgt_batch.size(1), beam_width)
                return predicted
        else:
            # decoding - regular
            for t in range(tgt_batch.size(1)):
                dec_in, hidden = self.decoder(dec_in, hidden, enc_results)
                if mode == DecodeMode.TRAIN:
                    batch_loss += self.criterion(dec_in, tgt_batch[:, t],
                                                 reduction='sum', ignore_index=iwslt.PAD_IDX)
                elif mode == DecodeMode.EVAL:
                    batch_loss += self.criterion(dec_in, tgt_batch[:, t],
                                                 reduction='sum', ignore_index=iwslt.PAD_IDX).item()
                if teacher_forcing and mode != DecodeMode.TRANSLATE:
                    dec_in = tgt_batch[:, t].unsqueeze(1)
                else:
                    topv, topi = dec_in.topk(1)
                    dec_in = topi.detach()
                if mode == DecodeMode.TRANSLATE:
                    predicted.append(dec_in.item())
                    if dec_in.item() == iwslt.EOS_IDX:
                        break
        # return results
        if mode == DecodeMode.TRANSLATE or mode == DecodeMode.TRANSLATE_BEAM:
            return predicted
        else:
            batch_loss /= tgt_batch.data.gt(0).sum().float()
            if mode == DecodeMode.EVAL:
                batch_loss = batch_loss.item()
            return batch_loss
