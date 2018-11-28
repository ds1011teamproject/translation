"""
super class for all of the various model we will build

Notes:
- the model should implement its own training loop / early stop
- it should track its own training curve
- it should track its own results

"""
import logging
import torch
import gc
import random
import numpy as np

from libs.models.BaseModel import BaseModel
import libs.data_loaders.IwsltLoader as iwslt
from libs.data_loaders.IwsltLoader import SRC, TAR, DataSplitType
from config.constants import (HyperParamKey, LoaderParamKey, ControlKey,
                              PathKey, StateKey, LoadingKey, OutputKey)
from libs.common.BleuScorer import BLEUScorer

logger = logging.getLogger('__main__')
BLEUscorer = BLEUScorer()


class DecodeMode:
    TRAIN = 'train'
    EVAL = 'eval'
    TRANSLATE = 'translate'
    TRANSLATE_BEAM = 'trans_beam'


class MTBaseModel(BaseModel):
    VAL_BLEU = 'val_BLEU'

    def __init__(self, hparams, lparams, cparams, label='scratch', nolog=False):
        super().__init__(hparams, lparams, cparams, label, nolog)

        self.encoder = None
        self.decoder = None
        self.enc_optim = None
        self.dec_optim = None
        self.enc_scheduler = None
        self.dec_scheduler = None
        self.criterion = None

        # todo: might add other metrics later
        self.iter_curves = {
            # self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            # self.VAL_ACC: [],
            self.VAL_LOSS: [],
            self.VAL_BLEU: []
        }
        # todo
        self.epoch_curves = {
            # self.TRAIN_ACC: [],
            self.TRAIN_LOSS: [],
            # self.VAL_ACC: [],
            self.VAL_LOSS: [],
            self.VAL_BLEU: []
        }

    def train(self, loader, tqdm_handler):
        if self.encoder is None or self.decoder is None:
            logger.error("Model not properly initialized! Stopping training on model {}".format(self.label))
        else:
            # init optim/scheduler/criterion
            self._init_optim_and_scheduler()
            self.criterion = self.hparams[HyperParamKey.CRITERION]
            early_stop = False
            best_loss = np.Inf
            best_bleu = 0
            ni_buffer = 0  # buffer for no improvement LR decay

            for epoch in tqdm_handler(range(self.hparams[HyperParamKey.NUM_EPOCH] - self.cur_epoch)):
                # lr_scheduler step
                self.enc_scheduler.step(epoch=self.cur_epoch)
                self.dec_scheduler.step(epoch=self.cur_epoch)
                self.cur_epoch += 1
                logger.info("stepped scheduler to epoch = {}".format(self.enc_scheduler.last_epoch + 1))

                # mini-batch train
                for i, (src, tgt, src_lens, _) in enumerate(loader.loaders[DataSplitType.TRAIN]):
                    # tune to train mode
                    self.encoder.train()
                    self.decoder.train()
                    self.enc_optim.zero_grad()
                    self.dec_optim.zero_grad()
                    # encoding
                    enc_results = self.encoder(src, src_lens)
                    # enc_results: for simple GRU: encoder context vector, for GRU_attention: (enc_out, hidden)
                    # decoding
                    teacher_forcing = True if random.random() < self.hparams[HyperParamKey.TEACHER_FORCING_RATIO] else False
                    batch_loss = self.decoding(tgt, enc_results, teacher_forcing, mode=DecodeMode.TRAIN)
                    # optimization step
                    batch_loss.backward()
                    self.enc_optim.step()
                    self.dec_optim.step()
                    # report/save/early-stop
                    if i % self.hparams[HyperParamKey.TRAIN_LOOP_EVAL_FREQ] == 0:
                        # a) compute losses
                        # train_loss = self.compute_loss(loader.loaders[DataSplitType.TRAIN])
                        train_loss = batch_loss
                        val_loss = self.compute_loss(loader.loaders[DataSplitType.VAL])
                        val_bleu = self.eval_model(loader)
                        # b) report
                        logger.info("(epoch){}/{} (step){}/{} (trainLoss){} (valLoss){} (valBLEU){} (lr)e:{}/d:{}".format(
                            self.cur_epoch, self.hparams[HyperParamKey.NUM_EPOCH],
                            i + 1, len(loader.loaders[DataSplitType.TRAIN]),
                            train_loss, val_loss, val_bleu,
                            self.enc_optim.param_groups[0]['lr'], self.dec_optim.param_groups[0]['lr']))
                        self.eval_randomly(loader.loaders[DataSplitType.TRAIN], loader.id2token[iwslt.TAR], 'Train')
                        self.eval_randomly(loader.loaders[DataSplitType.VAL], loader.id2token[iwslt.TAR], 'Val')
                        gc.collect()
                        # c) record current loss
                        self.iter_curves[self.TRAIN_LOSS].append(train_loss)
                        self.iter_curves[self.VAL_LOSS].append(val_loss)
                        self.iter_curves[self.VAL_BLEU].append(val_bleu)
                        # d) save if best
                        if val_loss < best_loss:
                            # update output_dict
                            self.output_dict[OutputKey.BEST_VAL_LOSS] = val_loss
                            # save if needed
                            if self.cparams[ControlKey.SAVE_BEST_MODEL]:
                                self.save(fn=self.BEST_FN)
                            # update current best
                            best_loss = val_loss
                        if val_bleu > best_bleu:
                            # update output
                            self.output_dict[OutputKey.BEST_VAL_BLEU] = val_bleu
                            # update best
                            best_bleu = val_bleu
                        # e) check whether to decay the LR due to no-improvements seen in many steps
                        ni_buffer = self.check_for_no_improvement_decay(ni_buffer)
                        ni_buffer -= 1

                        # f) check early-stop
                        if self.hparams[HyperParamKey.CHECK_EARLY_STOP]:
                            early_stop = self.check_early_stop()
                        if early_stop:
                            logger.info("--- stopping training due to early stop ---")
                            break

                # eval per epoch
                train_loss = self.compute_loss(loader.loaders[DataSplitType.TRAIN])
                val_loss = self.compute_loss(loader.loaders[DataSplitType.VAL])
                val_bleu = self.eval_model(loader)
                self.epoch_curves[self.TRAIN_LOSS].append(train_loss)
                self.epoch_curves[self.VAL_LOSS].append(val_loss)
                self.epoch_curves[self.VAL_BLEU].append(val_bleu)
                if self.cparams[ControlKey.SAVE_EACH_EPOCH]:
                    self.save()
                if early_stop:  # nested loop
                    break

            # final loss evaluate:
            train_loss = self.compute_loss(loader.loaders[DataSplitType.TRAIN])
            val_loss = self.compute_loss(loader.loaders[DataSplitType.VAL])
            val_bleu = self.eval_model(loader)
            self.output_dict[OutputKey.FINAL_TRAIN_LOSS] = train_loss
            self.output_dict[OutputKey.FINAL_VAL_LOSS] = val_loss
            self.output_dict[OutputKey.FINAL_VAL_BLEU] = val_bleu
            logger.info("training completed, results collected...")

    def save(self, fn=BaseModel.CHECKPOINT_FN):
        state = {
            StateKey.MODEL_STATE: {'encoder': self.encoder.state_dict(),
                                   'decoder': self.decoder.state_dict()},
            StateKey.OPTIM_STATE: {'encoder': self.enc_optim.state_dict(),
                                   'decoder': self.dec_optim.state_dict()},
            StateKey.SCHED_STATE: {'encoder': self.enc_scheduler.state_dict(),
                                   'decoder': self.dec_scheduler.state_dict()},
            StateKey.HPARAMS: self.hparams,
            StateKey.CPARAMS: self.cparams,
            StateKey.LPARAMS: self.lparams,
            StateKey.ITER_CURVES: self.iter_curves,
            StateKey.EPOCH_CURVES: self.epoch_curves,
            StateKey.CUR_EPOCH: self.cur_epoch,
            StateKey.LABEL: self.label
        }

        self._save_checkpoint(state, fn)

    def load(self, loaded):
        """
        can load either the best model, the checkpoint or a specific path
        """
        # if path_to_model_ovrd is None:
        #     if which_model == LoadingKey.LOAD_BEST:
        #         path_to_model_ovrd = self.cparams[PathKey.MODEL_PATH] + self.BEST_FN
        #     else:
        #         path_to_model_ovrd = self.cparams[PathKey.MODEL_PATH] + self.CHECKPOINT_FN
        #
        # logger.info("loading checkpoint at {}".format(path_to_model_ovrd))
        # loaded = torch.load(path_to_model_ovrd)
        # todo: init model(enc/dec) after reading hparams from checkpoint

        # load encoder/decoder
        self.encoder.load_state_dict(loaded[StateKey.MODEL_STATE]['encoder'])
        self.decoder.load_state_dict(loaded[StateKey.MODEL_STATE]['decoder'])
        # load optimizers
        self._init_optim()
        self.enc_optim.load_state_dict(loaded[StateKey.OPTIM_STATE]['encoder'])
        self.dec_optim.load_state_dict(loaded[StateKey.OPTIM_STATE]['decoder'])
        # load lr_schedulers
        self._init_scheduler()
        self.enc_scheduler.load_state_dict(loaded[StateKey.SCHED_STATE]['encoder'])
        self.dec_scheduler.load_state_dict(loaded[StateKey.SCHED_STATE]['decoder'])
        # load parameters
        self.hparams = loaded[StateKey.HPARAMS]
        self.lparams = loaded[StateKey.LPARAMS]
        self.cparams = loaded[StateKey.CPARAMS]
        # load train history
        self.iter_curves = loaded[StateKey.ITER_CURVES]
        self.epoch_curves = loaded[StateKey.EPOCH_CURVES]
        self.cur_epoch = loaded[StateKey.CUR_EPOCH]
        self.label = loaded[StateKey.LABEL]

        logger.info("Successfully loaded checkpoint!")

    def _init_optim(self):
        op_constr = self.hparams[HyperParamKey.OPTIMIZER]
        self.enc_optim = op_constr(self.encoder.parameters(), lr=self.hparams[HyperParamKey.ENC_LR])
        self.dec_optim = op_constr(self.decoder.parameters(), lr=self.hparams[HyperParamKey.DEC_LR])

    def _init_scheduler(self):
        sche_constr = self.hparams[HyperParamKey.SCHEDULER]
        self.enc_scheduler = sche_constr(self.enc_optim, gamma=self.hparams[HyperParamKey.SCHEDULER_GAMMA])
        self.dec_scheduler = sche_constr(self.dec_optim, gamma=self.hparams[HyperParamKey.SCHEDULER_GAMMA])

    ################################
    # Override following if needed #
    ################################
    def decoding(self, *args, **kwargs):
        raise Exception("[decoding] should be override!")

    def check_for_no_improvement_decay(self, ni_buffer):
        """
        part of the training loop, check for whether we need to decay the learning rate due to no progress in a while
        :param ni_buffer: the counter (towards 0) for the last improvement seen iteration
        :return: ni_buffer, resets the buffer if we see improvement
        """
        no_improvement = self.check_no_improvement()
        if no_improvement and self.hparams[HyperParamKey.NO_IMPROV_LR_DECAY] < 1.0 and ni_buffer <= 0:
            # setting lr on the schedulers
            for j, base_lr in enumerate(self.enc_scheduler.base_lrs):
                self.enc_scheduler.base_lrs[j] = base_lr * self.hparams[HyperParamKey.NO_IMPROV_LR_DECAY]

            for j, base_lr in enumerate(self.dec_scheduler.base_lrs):
                self.dec_scheduler.base_lrs[j] = base_lr * self.hparams[HyperParamKey.NO_IMPROV_LR_DECAY]

            # setting lr on the optimizers
            for param_group, lr in zip(self.enc_scheduler.optimizer.param_groups, self.enc_scheduler.get_lr()):
                param_group['lr'] = lr

            for param_group, lr in zip(self.dec_scheduler.optimizer.param_groups, self.dec_scheduler.get_lr()):
                param_group['lr'] = lr

            logger.info('reducing encoder base_lr to %.5f since no improvement observed in %s steps' % (
                self.enc_scheduler.base_lrs[0],
                self.hparams[HyperParamKey.NO_IMPROV_LOOK_BACK]
            ))

            logger.info('reducing decoder base_lr to %.5f since no improvement observed in %s steps' % (
                self.dec_scheduler.base_lrs[0],
                self.hparams[HyperParamKey.NO_IMPROV_LOOK_BACK]
            ))

            ni_buffer = self.hparams[HyperParamKey.NO_IMPROV_LOOK_BACK]
        return ni_buffer

    def check_no_improvement(self):
        """ boolean of whether we did not improve in the last x interations, where x is a hparam """
        val_loss_curve = self.iter_curves[self.VAL_LOSS]
        t = self.hparams[HyperParamKey.NO_IMPROV_LOOK_BACK]

        logger.info("Checking for no improvement, val loss history is %s" % str(val_loss_curve))
        if len(val_loss_curve) >= t + 1 and min(val_loss_curve[-t:]) > val_loss_curve[-t - 1]:
            return True
        return False

    def check_early_stop(self):
        lookback = self.hparams[HyperParamKey.EARLY_STOP_LOOK_BACK]
        threshold = self.hparams[HyperParamKey.EARLY_STOP_REQ_PROG]
        loss_hist = self.iter_curves[self.VAL_LOSS]
        logger.info("Checking for early stop, val loss history is %s" % str(loss_hist))
        if len(loss_hist) > lookback + 1 and min(loss_hist[-lookback:]) > loss_hist[-lookback - 1] - threshold:
            return True
        return False

    def eval_model(self, dataloader, score_only=True, use_beam=False, beam_width=3):
        """
        Evaluate model by computing BLEU score on validation set.
        :param dataloader: dataloader from model-manager
        :param score_only: if True, only return BLEU score (float)
        :param use_beam: if True, use beam-search for translation
        :param beam_width: search width for each step
        :return: BLEU result, a float if score_only, else a BLEU score class
        """
        true = []
        pred = []
        id2token = dataloader.id2token[iwslt.TAR]
        with torch.no_grad():
            for src, tgt, slen, _ in dataloader.loaders[DataSplitType.VAL]:
                for idx in range(len(src)):
                    # get sample
                    src_i = src[idx].unsqueeze(0)
                    tgt_i = tgt[idx].unsqueeze(0)
                    slen_i = slen[idx].unsqueeze(0)
                    # tune to eval mode
                    self.encoder.eval()
                    self.encoder.eval()
                    # encoding
                    enc_result = self.encoder(src_i, slen_i)
                    # decoding
                    if use_beam:
                        predicted = self.decoding(tgt_i, enc_result, teacher_forcing=False,
                                                  mode=DecodeMode.TRANSLATE_BEAM, beam_width=beam_width)[0]
                    else:
                        predicted = self.decoding(tgt_i, enc_result, False, mode=DecodeMode.TRANSLATE)
                    # convert to strings
                    target = " ".join([id2token[e] for e in tgt_i.squeeze() if e != iwslt.PAD_IDX])
                    translated = " ".join([id2token[e] for e in predicted])
                    true.append(target)
                    pred.append(translated)
        return BLEUscorer.bleu(true, [pred], score_only=score_only)

    def compute_loss(self, loader):
        """
        This computation is very time-consuming, slows down the training.
        (?) Solution: a) use large check_interval (current)
                      b) compute the loss on train/val loader only per epoch
        """
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            loss = 0
            for i, (src, tgt, slen, tlen) in enumerate(loader):
                # encoding
                enc_results = self.encoder(src, slen)
                # decoding
                teacher_forcing = True if random.random() < self.hparams[HyperParamKey.TEACHER_FORCING_RATIO] else False
                batch_loss = self.decoding(tgt, enc_results, teacher_forcing, mode=DecodeMode.EVAL)
                loss += batch_loss
        # normalize
        loss /= len(loader)
        return loss

    def eval_randomly(self, loader, id2token, loader_label):
        """Randomly translate a sentence from the given data loader"""
        with torch.no_grad():
            src, tgt, slen, _ = next(iter(loader))
            idx = random.choice(range(len(src)))
            src = src[idx].unsqueeze(0)
            tgt = tgt[idx].unsqueeze(0)
            slen = slen[idx].unsqueeze(0)
            self.encoder.eval()
            self.decoder.eval()
            # encoding
            enc_results = self.encoder(src, slen)
            # decoding
            predicted = self.decoding(tgt, enc_results, teacher_forcing=False, mode=DecodeMode.TRANSLATE_BEAM)[0]
        target = " ".join([id2token[e.item()] for e in tgt.squeeze() if e.item() != iwslt.PAD_IDX])
        translated = " ".join([id2token[e] for e in predicted])
        logger.info("Translate randomly from {}:\nTruth:{}\nPredicted:{}".format(loader_label, target, translated))
