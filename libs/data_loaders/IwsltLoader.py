"""
DataLoader for IWSLT data set.
"""
import os
import logging
import numpy as np
from collections import Counter
import pickle as pkl

from libs.data_loaders.BaseLoader import BaseLoader
from config.constants import (HyperParamKey as hparamKey, PathKey,
                              LoaderParamKey as loaderKey)
from config.basic_conf import DEVICE

logger = logging.getLogger('__main__')


##################
# IWSLT specific #
##################
SOS_TOKEN, SOS_IDX = '<SOS>', 0
EOS_TOKEN, EOS_IDX = '<EOS>', 1
UNK_TOKEN, UNK_IDX = '<UNK>', 2


class Language:
    VIET = 'vi'
    CHIN = 'zh'
    ENG = 'en'


class DataSplitType:
    TRAIN = 'train'
    VAL = 'dev'
    TEST = 'test'


#################
# IWSLT Classes #
#################
class IwsltLoader(BaseLoader):
    def __init__(self, cparams, hparams, tqdm):
        super().__init__(cparams, hparams, tqdm)
        pass

    def load(self):
        self._load_raw_data()
        self._data_to_pipe()
        # todo: convert index vectors to tensor? too much memory; or convert each datum in train loop?
        return {loaderKey.ACT_VOCAB_SIZE: self.hparams[hparamKey.VOC_SIZE]}

    def _load_raw_data(self):
        """
        Data preprocessing.
        Convert raw text from file into train/val/test data sets.
        """
        logger.info("Get source language datum list...")
        self.data['source'] = load_datum_list(data_path=self.cparams[PathKey.DATA_PATH],
                                                             lang=self.cparams[PathKey.INPUT_LANG])
        logger.info("Get target language datum list...")
        self.data['target'] = load_datum_list(data_path=self.cparams[PathKey.DATA_PATH],
                                                                lang=self.cparams[PathKey.OUTPUT_LANG])
        # get language vocabulary
        stoken2id_file = 'data/{}_indexer_voc{}.p'.format(self.cparams[PathKey.INPUT_LANG],
                                                          self.hparams[hparamKey.VOC_SIZE])
        ttoken2id_file = 'data/{}_indexer_voc{}.p'.format(self.cparams[PathKey.OUTPUT_LANG],
                                                          self.hparams[hparamKey.VOC_SIZE])
        try:
            stoken2id = pkl.load(open(stoken2id_file, 'rb'))
            ttoken2id = pkl.load(open(ttoken2id_file, 'rb'))
            logger.info("Language indexer found and loaded!")
        except IOError:
            stoken2id, sid2token, svocab = get_vocabulary(self.data['source'][0], vocab_size=self.hparams[hparamKey.VOC_SIZE])
            pkl.dump(stoken2id, open(stoken2id_file, 'wb'))
            ttoken2id, tid2token, tvocab = get_vocabulary(self.data['target'][0], vocab_size=self.hparams[hparamKey.VOC_SIZE])
            pkl.dump(ttoken2id, open(ttoken2id_file, 'wb'))
            logger.info("Generated indexer for both src/target languages!")
        # convert tokens to indices
        logger.info("Convert token to index for source language ...")
        self._update_datum_indices(stoken2id, mode='source')
        logger.info("Convert token to index for target language ...")
        self._update_datum_indices(ttoken2id, mode='target')

    def _update_datum_indices(self, indexer, mode='source'):
        datum_sets = self.data['source'] if mode == 'source' else self.data['target']
        for datum_set in datum_sets:  # train, val, test sets
            for datum in datum_set:
                datum.set_token_indices(
                    [indexer[tok] if tok in indexer else UNK_IDX for tok in datum.tokens]
                )

    def _data_to_pipe(self):
        pass


class IWSLTDatum:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.tokens = None
        self.token_indices = None

    def set_tokens(self, tokens):
        self.tokens = tokens

    def set_token_indices(self, indices):
        self.token_indices = indices


##################
# Util functions #
##################
def tokenize(line):
    """Simple split for using pre-tokenized data"""
    return line.replace("\n", "").split(" ")


def raw_to_datumlist(data_path, language, data_split_type):
    """
    Convert raw text from file into a Datum List
    :param data_path: path to find data file (tokenized iwslt data)
    :param language: data set language
    :param data_split_type: value of DataSplitType
    :return: list of IWSLTDatum
    """
    datum_list = []
    file_path = '{}{}.tok.{}'.format(data_path, data_split_type, language)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datum = IWSLTDatum(line)
            datum.set_tokens(tokenize(line))
            datum_list.append(datum)
    return datum_list


def load_datum_list(data_path, lang):
    return (raw_to_datumlist(data_path, lang, DataSplitType.TRAIN),
            raw_to_datumlist(data_path, lang, DataSplitType.VAL),
            raw_to_datumlist(data_path, lang, DataSplitType.TEST))


def get_vocabulary(datum_list, vocab_size):
    """
    Generate token2id, id2token, vocabulary
    """
    vocab = [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    word_counter = Counter()
    for datum in datum_list:
        word_counter.update(Counter(datum.tokens))
    vocab += [d[0] for d in word_counter.most_common(vocab_size - 3)]  # save 3 places for SOS/EOS/UNK
    token2id = dict([(tok, vocab.index(tok)) for tok in vocab])
    id2token = dict([(vocab.index(tok), tok) for tok in vocab])
    return token2id, id2token, vocab
