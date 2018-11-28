"""
DataLoader for IWSLT data set.

**NOTE**:
The collate function automatically sorts by length of the source sentence!
"""

from collections import Counter
import io
import math
import numpy as np
import pickle as pkl
import random

import torch
from torch.utils.data import DataLoader, Dataset

from config.basic_conf import DEVICE
from config.constants import (HyperParamKey as hparamKey, PathKey,
                              LoaderParamKey as loaderKey)
from libs.data_loaders.BaseLoader import BaseLoader

import logging
logger = logging.getLogger('__main__')


##################
# IWSLT specific #
##################
PAD_TOKEN, PAD_IDX = '<PAD>', 0
UNK_TOKEN, UNK_IDX = '<UNK>', 1
SOS_TOKEN, SOS_IDX = '<SOS>', 2
EOS_TOKEN, EOS_IDX = '<EOS>', 3

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
    def __init__(self, cparams, hparams):
        super().__init__(cparams, hparams)
        self.token2id = {}
        self.id2token = {}
        self.trained_emb = {}

    def load(self):
        self._load_raw_data()
        self._data_to_pipe()
        results = {loaderKey.ACT_VOCAB_SIZE: {
            SRC: len(self.token2id[SRC]),
            TAR: len(self.token2id[TAR])}}
        # if self.hparams[hparamKey.USE_FT_EMB]:
        #     results[loaderKey.TRAINED_EMB] = {
        #         SRC: self.trained_emb[SRC],
        #         TAR: self.trained_emb[TAR]}
        return results

    def _load_raw_data(self):
        """
        Data preprocessing.
        Convert raw text from file into train/val/test data sets.
        """
        logger.info("Get source language datum list...")
        data_path = self.cparams[PathKey.DATA_PATH] + \
            'iwslt-%s-en/' % self.cparams[PathKey.INPUT_LANG]
        self.data[SRC] = load_datum_list(
            data_path=data_path,
            lang=self.cparams[PathKey.INPUT_LANG],
            num_to_load=self.hparams[hparamKey.NUM_TRAIN_SENT_TO_LOAD])
        logger.info("Get target language datum list...")
        self.data[TAR] = load_datum_list(
            data_path=data_path,
            lang=self.cparams[PathKey.OUTPUT_LANG],
            num_to_load=self.hparams[hparamKey.NUM_TRAIN_SENT_TO_LOAD])
        # get language vocabulary
        svocab_file = 'data/{}_voc{}.p'.format(self.cparams[PathKey.INPUT_LANG],
                                               self.hparams[hparamKey.VOC_SIZE])
        tvocab_file = 'data/{}_voc{}.p'.format(self.cparams[PathKey.OUTPUT_LANG],
                                               self.hparams[hparamKey.VOC_SIZE])
        try:
            svocab = pkl.load(open(svocab_file, 'rb'))
            tvocab = pkl.load(open(tvocab_file, 'rb'))
            logger.info(
                "Vocabulary found and loaded! (token2id, id2token, vocabs)")
        except IOError:
            logger.info(
                "Building Vocabulary from train set ... building source vocab")
            svocab = get_vocabulary(self.data[SRC][DataSplitType.TRAIN],
                                    self.hparams[hparamKey.VOC_SIZE])
            logger.info("Building target vocab")
            tvocab = get_vocabulary(self.data[TAR][DataSplitType.TRAIN],
                                    self.hparams[hparamKey.VOC_SIZE])
            # save to file
            pkl.dump(svocab, open(svocab_file, 'wb'))
            pkl.dump(tvocab, open(tvocab_file, 'wb'))
            logger.info(
                "Generated token2id, id2token for both src/target languages!")

        # keep token2id, id2token in memory
        self.token2id[SRC] = svocab['token2id']
        self.token2id[TAR] = tvocab['token2id']
        self.id2token[SRC] = svocab['id2token']
        self.id2token[TAR] = tvocab['id2token']

        # Word embeddings
        if self.hparams[hparamKey.USE_FT_EMB]:
            logger.info("Loading fastText embeddings ...")
            ft_path = self.cparams[PathKey.DATA_PATH] + 'word_vectors/'
            logger.info("Loading source embeddings...")
            self.trained_emb[SRC] = get_fasttext_embedding(
                ft_path,
                self.cparams[PathKey.INPUT_LANG],
                svocab['id2token'],
                self.hparams[hparamKey.EMBEDDING_DIM],
            )
            logger.info("Loading target embeddings...")
            self.trained_emb[TAR] = get_fasttext_embedding(
                ft_path,
                self.cparams[PathKey.OUTPUT_LANG],
                tvocab['id2token'],
                self.hparams[hparamKey.EMBEDDING_DIM],
            )
        else:
            self.trained_emb[SRC] = None
            self.trained_emb[TAR] = None

        # convert tokens to indices
        logger.info("Convert token to index for source language ...")
        self._update_datum_indices(self.token2id[SRC], mode=SRC)
        logger.info("Convert token to index for target language ...")
        self._update_datum_indices(self.token2id[TAR], mode=TAR)
        logger.info("Datum list loaded for both src/target languages!")

    def _update_datum_indices(self, indexer, mode=SRC):
        datum_sets = self.data[SRC] if mode == SRC else self.data[TAR]
        for split in datum_sets:  # train, val, test
            for datum in datum_sets[split]:
                datum.set_token_indices(
                    [indexer[tok] if tok in indexer
                     else UNK_IDX for tok in datum.tokens] +
                    [EOS_IDX]
                )  # add EOS at the end of the sentence

    def _data_to_pipe(self):
        """
        coverts the data objects to the torch.*.DataLoader pipes
        """
        logger.info("Loading raw data into the DataLoaders ...")
        shuffle_dict = {DataSplitType.TRAIN: False,
                        DataSplitType.VAL: False, DataSplitType.TEST: False}
        assert self.data[SRC].keys() == self.data[TAR].keys(), \
            "Source and Target data do not have the same keys!" \
            + "cannot construct DataLoaders"
        for split in self.data[SRC].keys():  # train, val, test
            cur_ds = IWSLTDataset(self.data[SRC][split], self.data[TAR][split],
                                  max_length=self.hparams[hparamKey.MAX_LENGTH],
                                  batch_size=self.hparams[hparamKey.BATCH_SIZE])
            self.loaders[split] = DataLoader(
                dataset=cur_ds,
                batch_size=self.hparams[hparamKey.BATCH_SIZE],
                collate_fn=self.iwslt_collate_func,
                shuffle=shuffle_dict[split])

    def iwslt_collate_func(self, batch):
        """
        **NOTE**:
        The batch is automatically sorted by length of the source sentence!
        """
        s_list, t_list = [], []  # source list and target list of token indices
        s_len_list, t_len_list = [], []  # source and target list of lengths
        for datum in batch:
            s_len_list.append(datum[2])
            t_len_list.append(datum[3])

        max_length1 = self.hparams[hparamKey.MAX_LENGTH]  # np.max(s_len_list)
        max_length2 = self.hparams[hparamKey.MAX_LENGTH]  # np.max(t_len_list)

        # padding
        for datum in batch:
            padded_vec1 = np.pad(np.array(datum[0]),
                                 pad_width=(0, max_length1 - datum[2]),
                                 mode="constant", constant_values=0)
            s_list.append(padded_vec1)

            padded_vec2 = np.pad(np.array(datum[1]),
                                 pad_width=(0, max_length2 - datum[3]),
                                 mode="constant", constant_values=0)
            t_list.append(padded_vec2)

        n1 = np.array(s_list).astype(int)
        n2 = np.array(t_list).astype(int)

        t_sent1 = torch.from_numpy(n1).to(DEVICE)
        t_sent2 = torch.from_numpy(n2).to(DEVICE)
        t_len1 = torch.LongTensor(s_len_list).to(DEVICE)
        t_len2 = torch.LongTensor(t_len_list).to(DEVICE)

        # sorting by descending len1
        sorted_t_len1, idx_sort = torch.sort(t_len1, dim=0, descending=True)

        return [
            torch.index_select(t_sent1, 0, idx_sort),
            torch.index_select(t_sent2, 0, idx_sort),
            sorted_t_len1,
            torch.index_select(t_len2, 0, idx_sort),
        ]


class IWSLTDatum:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.tokens = None
        self.token_indices = None
        self.num_toks = 0

    def set_tokens(self, tokens):
        self.tokens = tokens
        self.num_toks = len(tokens)

    def set_token_indices(self, indices):
        self.token_indices = indices


class IWSLTDataset(Dataset):
    def __init__(self, source_list, target_list, max_length, batch_size):
        assert len(source_list) == len(target_list), \
            "Length of source and target is not the same!" \
            + " Cannot construct Dataset object"

        # Prepare batch data (sort and group by sentence length)
        source_list, target_list = prepare_batch_data(
            source_list, target_list, batch_size)

        self.s_list = source_list
        self.t_list = target_list
        self.max_length = max_length

    def __len__(self):
        # already asserted that the 2 lengths are the same (source/target)
        return len(self.s_list)

    def __getitem__(self, idx):
        """Get indices vector for i-th IWSLT Datum
        order is (source, target, source len, target len)
        """
        src = self.s_list[idx].token_indices[:self.max_length]
        tgt = self.t_list[idx].token_indices[:self.max_length]
        return [src, tgt, len(src), len(tgt)]


##################
# Util functions #
##################
def tokenize(line):
    """Simple split for using pre-tokenized data"""
    if line == '':
        return []
    line = line[0].lower() + line[1:]
    return line.replace("\n", "").split()


def raw_to_datumlist(data_path, language, data_split_type, num_to_load=None):
    """
    Convert raw text from file into a Datum List
    :param data_path: path to find data file (tokenized iwslt data)
    :param language: data set language
    :param data_split_type: value of DataSplitType
    :return: list of IWSLTDatum
    """
    datum_list = []
    file_path = '{}{}.tok.{}'.format(data_path, data_split_type, language)
    lines_loaded = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datum = IWSLTDatum(line)
            datum.set_tokens(tokenize(line))
            datum_list.append(datum)
            lines_loaded += 1

            if num_to_load is not None and num_to_load > 0:
                if lines_loaded >= num_to_load:
                    break

    return datum_list


def load_datum_list(data_path, lang, num_to_load=None):
    return {
        DataSplitType.TRAIN: raw_to_datumlist(
            data_path, lang, DataSplitType.TRAIN, num_to_load),
        DataSplitType.VAL: raw_to_datumlist(
            data_path, lang, DataSplitType.VAL, num_to_load),
        DataSplitType.TEST: raw_to_datumlist(
            data_path, lang, DataSplitType.TEST, num_to_load)}


def prepare_batch_data(source_list, target_list, batch_size):
    """
    Prepare input data for batch processing:
        - filter empty sentence pairs
        - sort aligned IWSLTDatum() data by sentence length
        - group sentences of similar length (based on batch size)
        - shuffle even-sized groups
    """
    # Create aligned tuples of (source, target)
    tup = list(zip(source_list, target_list))

    # Filter empty sentences
    tup = [pair for pair in tup if
           ((pair[0].num_toks > 0) and (pair[1].num_toks > 0))]

    # Sort tuples by number of tokens in source sentence
    tup_sort = sorted(tup, key=lambda pair: pair[0].num_toks)

    # Split data into evenly sized chunks
    n = math.ceil(len(source_list) / batch_size)  # Round up
    chunks = [tup_sort[i:i + n] for i in range(0, len(tup_sort), n)]

    # Split chunks into even-sized chunks (shuffle) and last chunk (no shuffle)
    even_chunks = chunks[:-1]
    last_chunk = chunks[-1]
    random.shuffle(even_chunks)  # Shuffle even chunks
    chunks = even_chunks + [last_chunk]  # Merge chunks

    # Flatten chunk sublists
    flatten = [pair for chunk in chunks for pair in chunk]
    # Unpack flat list of tuples
    source_list, target_list = zip(*flatten)

    return source_list, target_list


def get_vocabulary(datum_list, vocab_size):
    """
    Generate token2id, id2token, vocabulary
    """
    vocab = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    token2id = {}

    word_counter = Counter()
    for datum in datum_list:
        word_counter.update(Counter(datum.tokens))
    vocab += [d[0] for d in word_counter.most_common(vocab_size)]
    for i, word in enumerate(vocab):
        token2id[word] = i
    return {'token2id': token2id,
            'id2token': vocab}


def get_fasttext_embedding(data_path, language, id2token, emb_dim):
    """
    Load pre-trained word embeddings
    """
    ft_file = '{}cc.{}.300.vec'.format(data_path, language)

    # Load fastText embeddings for vocabulary words
    toks = set(id2token)
    fin = io.open(
        ft_file, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        if line.split()[0] in toks:
            tokens = line.rstrip().split(" ")
            data[tokens[0]] = list(map(float, tokens[1:]))

    # Initialize empty weights matrix
    weights = np.zeros((len(toks), 300))
    found = 0

    for i, tok in enumerate(id2token):
        try:
            weights[i] = data[tok]
            found += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    return weights
