"""
DataLoader for IWSLT data set.
"""
import os
import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset
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

SRC = 'source'
TAR = 'target'

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
        self.token2id = {}
        self.id2token = {}
        pass

    def load(self):
        self._load_raw_data()
        self._data_to_pipe()
        return {loaderKey.ACT_VOCAB_SIZE: {SRC: len(self.token2id[SRC]),
                                           TAR: len(self.token2id[TAR])}}

    def _load_raw_data(self):
        """
        Data preprocessing.
        Convert raw text from file into train/val/test data sets.
        """
        logger.info("Get source language datum list...")
        self.data[SRC] = load_datum_list(data_path=self.cparams[PathKey.DATA_PATH],
                                         lang=self.cparams[PathKey.INPUT_LANG])
        logger.info("Get target language datum list...")
        self.data[TAR] = load_datum_list(data_path=self.cparams[PathKey.DATA_PATH],
                                         lang=self.cparams[PathKey.OUTPUT_LANG])
        # get language vocabulary
        svocab_file = 'data/{}_voc{}.p'.format(self.cparams[PathKey.INPUT_LANG],
                                               self.hparams[hparamKey.VOC_SIZE])
        tvocab_file = 'data/{}_voc{}.p'.format(self.cparams[PathKey.OUTPUT_LANG],
                                               self.hparams[hparamKey.VOC_SIZE])
        try:
            svocab = pkl.load(open(svocab_file, 'rb'))
            tvocab = pkl.load(open(tvocab_file, 'rb'))
            logger.info("Vocabulary found and loaded! (token2id, id2token, vocabs)")
        except IOError:
            # build vocabulary
            svocab = get_vocabulary(self.data[SRC][0], vocab_size=self.hparams[hparamKey.VOC_SIZE])
            tvocab = get_vocabulary(self.data[TAR][0], vocab_size=self.hparams[hparamKey.VOC_SIZE])
            # save to file
            pkl.dump(svocab, open(svocab_file, 'wb'))
            pkl.dump(tvocab, open(tvocab_file, 'wb'))
            logger.info("Generated token2id, id2token for both src/target languages!")
        # keep token2id, id2token in memory
        self.token2id[SRC] = svocab['token2id']
        self.token2id[TAR] = tvocab['token2id']
        self.id2token[SRC] = svocab['id2token']
        self.id2token[TAR] = tvocab['id2token']
        # convert tokens to indices
        logger.info("Convert token to index for source language ...")
        self._update_datum_indices(self.token2id[SRC], mode=SRC)
        logger.info("Convert token to index for target language ...")
        self._update_datum_indices(self.token2id[TAR], mode=TAR)
        logger.info("Datum list loaded for both src/target languages!")

    def _update_datum_indices(self, indexer, mode=SRC):
        datum_sets = self.data[SRC] if mode == SRC else self.data[TAR]
        for datum_set in datum_sets:  # train, val, test sets
            for datum in datum_set:
                datum.set_token_indices(
                    [indexer[tok] if tok in indexer else UNK_IDX for tok in datum.tokens] + [EOS_IDX]
                )  # add EOS at the end of the sentence

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


# todo: pytorch Dataset for IWSLT data
class IWSLTDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get indices vector for i-th IWSLT Datum"""
        return self.data_list[idx].token_indices


##################
# Util functions #
##################
def tokenize(line):
    """Simple split for using pre-tokenized data"""
    if line == '':
        return []
    line = line[0].lower() + line[1:]
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
    vocab += [d[0] for d in word_counter.most_common(vocab_size)]
    token2id = dict([(tok, vocab.index(tok)) for tok in vocab])
    id2token = dict([(vocab.index(tok), tok) for tok in vocab])
    return {'token2id': token2id,
            'id2token': id2token,
            'vocab': vocab}


# todo: collate function for IWSLTDataset
def iwslt_collate_func(batch):
    pass
