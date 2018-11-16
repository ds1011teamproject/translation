"""
Basic Seq2Seq (Encoder-Decoder) model for translation.
"""
import logging
import random
import torch
import torch.nn.functional as F
import numpy as np

from libs.models.TranslatorModel import MTBaseModel
from libs.models.modules import GRU
import libs.data_loaders.IwsltLoader as iwslt
from libs.data_loaders.IwsltLoader import SRC, TAR, DataSplitType
from config.constants import (HyperParamKey as hparamKey, LoaderParamKey as loaderKey,
                              PathKey, StateKey, LoadingKey, ControlKey, OutputKey)
from config.basic_conf import DEVICE

logger = logging.getLogger('__main__')


class RNN_GRU(MTBaseModel):

    def __init__(self, hparams, lparams, cparams, label='gru-scratch', nolog=True):
        super().__init__(hparams, lparams, cparams, label, nolog)
        # encoder-decoder system
        self.encoder = GRU.Encoder(vocab_size=self.lparams[loaderKey.ACT_VOCAB_SIZE][SRC],  # source language vocab size
                                   emb_size=self.hparams[hparamKey.EMBEDDING_DIM],
                                   hidden_size=self.hparams[hparamKey.HIDDEN_SIZE],
                                   num_layers=self.hparams[hparamKey.ENC_NUM_LAYERS],
                                   num_directions=self.hparams[hparamKey.ENC_NUM_DIRECTIONS],
                                   dropout_prob=self.hparams.get(hparamKey.ENC_DROPOUT, 0.0),
                                   trained_emb=self.lparams[loaderKey.TRAINED_EMB][SRC] if self.hparams[hparamKey.USE_FT_EMB] else None
                                   ).to(DEVICE)
        self.decoder = GRU.Decoder(vocab_size=self.lparams[loaderKey.ACT_VOCAB_SIZE][TAR],  # target language vocab size
                                   emb_size=self.hparams[hparamKey.EMBEDDING_DIM],
                                   hidden_size=self.hparams[hparamKey.HIDDEN_SIZE],
                                   num_layers=self.hparams[hparamKey.DEC_NUM_LAYERS],
                                   num_directions=self.hparams[hparamKey.DEC_NUM_DIRECTIONS],
                                   dropout_prob=self.hparams.get(hparamKey.DEC_DROPOUT, 0.0),
                                   trained_emb=self.lparams[loaderKey.TRAINED_EMB][TAR] if self.hparams[hparamKey.USE_FT_EMB] else None
                                   ).to(DEVICE)

    def eval_model(self, dataloader):
        pass

    def compute_loss(self, loader, criterion):
        """
        This computation is very time-consuming, slows down the training.
        (?) Solution: a) use large check_interval (current)
                      b) compute the loss on train/val loader only per epoch
        """
        self.encoder.eval()
        self.decoder.eval()
        loss = 0
        for i, (src, tgt, slen, tlen) in enumerate(loader):
            batch_loss = 0
            batch_size = src.size()[0]
            # encoding
            enc_last_hidden = self.encoder(src, slen)
            # decoding
            dec_in = torch.LongTensor([iwslt.SOS_IDX] * batch_size).unsqueeze(1).to(DEVICE)
            for t in range(tgt.size(1)):  # seq_len axis
                dec_out = self.decoder(dec_in, enc_last_hidden)
                batch_loss += criterion(dec_out, tgt[:, t], reduction='sum',
                                        ignore_index=iwslt.PAD_IDX)
                topv, topi = dec_out.topk(1)
                dec_in = topi.detach()
            batch_loss /= tgt.data.gt(0).sum().float()
            loss += batch_loss.item()
        # normalize
        loss /= len(loader)
        return loss

    def eval_randomly(self, loader, id2token):
        """Randomly translate a sentence from the given data loader"""
        src, tgt, slen, _ = next(iter(loader))
        idx = random.choice(range(len(src)))
        src = src[idx].unsqueeze(0)
        tgt = tgt[idx].unsqueeze(0)
        slen = slen[idx].unsqueeze(0)
        self.encoder.eval()
        self.decoder.eval()
        enc_hidden = self.encoder(src, slen)
        predicted = []
        dec_in = torch.LongTensor([iwslt.SOS_IDX]).unsqueeze(1).to(DEVICE)
        for t in range(tgt.size(1)):
            dec_out = self.decoder(dec_in, enc_hidden)
            topv, topi = dec_out.topk(1)
            dec_in = topi.detach()
            predicted.append(dec_in)
            if dec_in.item() == iwslt.EOS_IDX:
                break
        target = " ".join([id2token[e.item()] for e in tgt.squeeze() if e.item() != iwslt.PAD_IDX])
        translated = " ".join([id2token[e.item()] for e in predicted])
        logger.info("Translate randomly selected sentence:\nTruth:{}\nPredicted:{}".format(target, translated))

    def check_early_stop(self):
        lookback = self.hparams[hparamKey.EARLY_STOP_LOOK_BACK]
        threshold = self.hparams[hparamKey.EARLY_STOP_REQ_PROG]
        loss_hist = self.iter_curves[self.TRAIN_LOSS]
        if len(loss_hist) > lookback + 1 and min(loss_hist[-lookback:]) > loss_hist[-lookback-1] - threshold:
            return True
        return False

    def train(self, loader, tqdm_handler):
        # todo: different from classification model train
        # todo: override train() from BaseModel
        if self.encoder is None or self.decoder is None:
            logger.error("Model not properly initialized! Stopping training on model {}".format(self.label))
        else:
            # init optim/scheduler/criterion
            self._init_optim_and_scheduler()
            # todo: set criterion in hyper-parameter config
            criterion = self.hparams[hparamKey.CRITERION]  # with brackets? "()"
            early_stop = False
            best_loss = np.Inf

            # epoch train
            for epoch in tqdm_handler(range(self.hparams[hparamKey.NUM_EPOCH] - self.cur_epoch)):
                # epoch_loss = 0
                # lr_scheduler step
                self.enc_scheduler.step(epoch=self.cur_epoch)
                self.dec_scheduler.step(epoch=self.cur_epoch)
                self.cur_epoch += 1
                logger.info("stepped scheduler to epoch = {}".format(self.enc_scheduler.last_epoch + 1))

                # mini-batch train
                # num_batch_trained = 0
                for i, (src, tgt, src_lens, tgt_lens) in enumerate(loader.loaders[DataSplitType.TRAIN]):
                    batch_loss = 0
                    num_batch_trained = i
                    # tune to train mode
                    self.encoder.train()
                    self.decoder.train()
                    self.enc_optim.zero_grad()
                    self.dec_optim.zero_grad()
                    batch_size = src.size()[0]
                    # encoding
                    enc_last_hidden = self.encoder(src, src_lens)
                    # decoding
                    dec_in = torch.LongTensor([iwslt.SOS_IDX] * batch_size).unsqueeze(1).to(DEVICE)
                    teacher_forcing = True if random.random() < self.hparams[hparamKey.TEACHER_FORCING_RATIO] else False
                    for t in range(tgt.size(1)):  # step through time/seq_len axis
                        dec_out = self.decoder(dec_in, enc_last_hidden)
                        batch_loss += criterion(dec_out, tgt[:, t], reduction='sum', ignore_index=iwslt.PAD_IDX)
                        # generate next dec_in
                        if teacher_forcing:
                            dec_in = tgt[:, t].unsqueeze(1)
                        else:
                            topv, topi = dec_out.topk(1)
                            dec_in = topi.detach()
                    # normalize loss by number of unpadded tokens
                    batch_loss /= tgt.data.gt(0).sum().float()
                    # optimization step
                    batch_loss.backward()
                    self.enc_optim.step()
                    self.dec_optim.step()
                    # update epoch_loss
                    # epoch_loss += batch_loss.item()
                    # report and check early-stop
                    if i % self.hparams[hparamKey.TRAIN_LOOP_EVAL_FREQ] == 0:
                        # a) compute losses
                        train_loss = self.compute_loss(loader.loaders[DataSplitType.TRAIN], criterion)
                        val_loss = self.compute_loss(loader.loaders[DataSplitType.VAL], criterion)
                        # b) report
                        logger.info("(epoch){}/{} (step){}/{} (trainLoss){} (valLoss){} (lr)e:{}/d:{}".format(
                            self.cur_epoch, self.hparams[hparamKey.NUM_EPOCH],
                            i + 1, len(loader.loaders[DataSplitType.TRAIN]),
                            train_loss, val_loss,
                            self.enc_optim.param_groups[0]['lr'], self.dec_optim.param_groups[0]['lr']))
                        # todo: eval randomly
                        self.eval_randomly(loader.loaders[DataSplitType.TRAIN], loader.id2token[iwslt.TAR])
                        # c) record current batch_loss
                        self.iter_curves[self.TRAIN_LOSS].append(train_loss)
                        self.iter_curves[self.VAL_LOSS].append(val_loss)
                        # d) save if best
                        if val_loss < best_loss:
                            # update output_dict
                            self.output_dict[OutputKey.BEST_VAL_LOSS] = val_loss
                            if self.cparams[ControlKey.SAVE_BEST_MODEL]:
                                self.save(fn=self.BEST_FN)
                            best_loss = val_loss
                        # e) check early-stop
                        if self.hparams[hparamKey.CHECK_EARLY_STOP]:
                            early_stop = self.check_early_stop()
                        if early_stop:
                            logger.info('--- stopping training due to early stop ---')
                            break
                # # normalize epoch loss
                # epoch_loss /= num_batch_trained  # not len(loader.loaders[TRAIN]) for early-stop

                # eval per epoch
                train_loss = self.compute_loss(loader.loaders[DataSplitType.TRAIN], criterion)
                val_loss = self.compute_loss(loader.loaders[DataSplitType.VAL], criterion)
                self.epoch_curves[self.TRAIN_LOSS].append(train_loss)
                self.epoch_curves[self.VAL_LOSS].append(val_loss)
                if self.cparams[ControlKey.SAVE_EACH_EPOCH]:
                    self.save()
                if early_stop:  # nested loop
                    break

            # final loss evaluate:
            train_loss = self.compute_loss(loader.loaders[DataSplitType.TRAIN], criterion)
            val_loss = self.compute_loss(loader.loaders[DataSplitType.VAL], criterion)
            self.output_dict[OutputKey.FINAL_TRAIN_LOSS] = train_loss
            self.output_dict[OutputKey.FINAL_VAL_LOSS] = val_loss
            logger.info("training completed, results collected...")
